"""Страница: Features & Clusters

Функциональность:
- Получение сводки кластеров и меток из API.
- Scatter UMAP/HDBSCAN (если есть проекция; иначе центроиды по кластерам).
- Выбор кластера, статистика, уведомление если кластера нет.
- Экспорт выбранного кластера в CSV.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import requests


def _api() -> str:
    # Единая логика: сначала API_URL (compose), затем API_BASE_URL (совместимость), затем сервисное имя
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def _auth_headers() -> Dict[str, str]:
    token = st.session_state.get("token") if isinstance(st.session_state, dict) else None
    return {"Authorization": f"Bearer {token}"} if token else {}


@st.cache_data(show_spinner=False, ttl=60)
def fetch_distribution(anomalies_only: bool = False) -> Dict[str, Any] | None:
    url = f"{_api()}/admin/clustering/distribution"
    try:
        r = requests.get(url, params={"anomalies_only": str(bool(anomalies_only)).lower()}, headers=_auth_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Ошибка загрузки распределения: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=60)
def fetch_labels() -> List[Dict[str, Any]]:
    url = f"{_api()}/admin/clustering/labels"
    try:
        r = requests.get(url, headers=_auth_headers(), timeout=30)
        if r.status_code == 200:
            return r.json() or []
    except Exception as e:
        st.warning(f"Ошибка загрузки меток кластеров: {e}")
    return []


def _build_scatter_data(dist: Dict[str, Any]) -> pd.DataFrame:
    # Пытаемся взять покомпонентные точки (reduced/points)
    # Ожидаемые поля: points: [{x,y,cluster_id,feature_id}], или reduced: [[x,y],...], labels: [...]
    if isinstance(dist, dict):
        points = dist.get("points") or []
        if isinstance(points, list) and points and all(isinstance(p, dict) for p in points):
            df = pd.DataFrame(points)
            # нормализуем имена
            if 'cluster' in df.columns and 'cluster_id' not in df.columns:
                df = df.rename(columns={'cluster': 'cluster_id'})
            return df
        # Альтернатива: reduced + labels
        reduced = dist.get("reduced")
        labels = dist.get("labels")
        feats = dist.get("feature_ids")
        if isinstance(reduced, list) and isinstance(labels, list):
            rows = []
            for i, coords in enumerate(reduced):
                x, y = (coords[0], coords[1]) if isinstance(coords, (list, tuple)) and len(coords) >= 2 else (None, None)
                rows.append({
                    'x': x,
                    'y': y,
                    'cluster_id': labels[i] if i < len(labels) else None,
                    'feature_id': (feats[i] if isinstance(feats, list) and i < len(feats) else None)
                })
            return pd.DataFrame(rows)
        # Последний вариант: берём центроиды кластеров как точки
        clusters = dist.get("clusters") or []
        if isinstance(clusters, list) and clusters and all(isinstance(c, dict) for c in clusters):
            rows = []
            for c in clusters:
                cid = c.get('cluster_id')
                size = c.get('count') or c.get('size')
                cx, cy = None, None
                centroid = c.get('centroid') or c.get('center') or {}
                if isinstance(centroid, dict):
                    cx, cy = centroid.get('x'), centroid.get('y')
                rows.append({'x': cx, 'y': cy, 'cluster_id': cid, 'count': size})
            return pd.DataFrame(rows)
    return pd.DataFrame()


def _legend_from_labels(labels: List[Dict[str, Any]]) -> Dict[int, str]:
    legend = {}
    for row in labels:
        try:
            cid = int(row.get('cluster_id'))
            name = row.get('defect_id') or row.get('description') or f"Кластер {cid}"
            legend[cid] = str(name)
        except Exception:
            continue
    return legend


def _export_cluster_csv(df_points: pd.DataFrame, cluster_id: int) -> bytes:
    cols = [c for c in ['feature_id', 'cluster_id', 'x', 'y'] if c in df_points.columns]
    data = df_points[df_points['cluster_id'] == cluster_id][cols].copy()
    return data.to_csv(index=False).encode('utf-8')


def render() -> None:
    st.title("🧭 Признаки и кластеры")
    st.caption("Признаки (RMS/FFT/статистики) и кластеры UMAP/HDBSCAN")

    # 1) Фильтры / Выбор
    with st.container(border=True):
        st.subheader("Фильтры")
        st.caption("Данные загружаются из административных эндпоинтов кластеризации")
        anomalies_only = st.checkbox("Показывать только аномальные окна", value=False)

    dist = fetch_distribution(anomalies_only=anomalies_only)
    labels = fetch_labels()
    if not dist:
        st.info("Кластеризация ещё не готова. Повторите позже или запустите пайплайн из админки.")
        return

    legend_map = _legend_from_labels(labels)

    # 2) Визуализация
    with st.container(border=True):
        st.subheader("Визуализация: кластерная проекция")
        df_points = _build_scatter_data(dist)
        if df_points.empty or df_points[['x', 'y']].isnull().all(axis=None):
            st.warning("Нет точек для проекции. Будут показаны только агрегированные сведения о кластерах.")
        else:
            # подписываем кластеры метками, если есть
            if 'cluster_id' in df_points.columns and not df_points.empty:
                df_points['cluster_name'] = df_points['cluster_id'].map(legend_map).fillna(df_points['cluster_id'].astype(str))
            fig = px.scatter(
                df_points,
                x='x', y='y',
                color=('cluster_name' if 'cluster_name' in df_points.columns else 'cluster_id'),
                hover_data=[c for c in ['feature_id', 'cluster_id'] if c in df_points.columns],
                title=None,
                color_discrete_sequence=["#003057", "#0095D9", "#5DC9F5", "#0B1F35", "#7FB3D5"],
            )
            fig.update_layout(plot_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)

    # Список кластеров
    clusters = dist.get('clusters') if isinstance(dist, dict) else None
    cluster_ids: List[int] = []
    if isinstance(clusters, list):
        for c in clusters:
            try:
                cid = int(c.get('cluster_id'))
                cluster_ids.append(cid)
            except Exception:
                continue
    elif 'labels' in (dist or {}):
        # соберём уникальные из labels
        try:
            cluster_ids = sorted({int(x) for x in dist.get('labels') if x is not None})
        except Exception:
            cluster_ids = []

    with st.container(border=True):
        st.subheader("Статистика кластера")
        if not cluster_ids:
            st.info("Кластера отсутствуют. Повторите операцию позже.")
        else:
            cid = st.selectbox("Выберите кластер", cluster_ids, format_func=lambda c: legend_map.get(c, f"Кластер {c}"))

            # Подсчёты
            count = None
            if isinstance(clusters, list):
                for c in clusters:
                    if c.get('cluster_id') == cid:
                        count = c.get('count') or c.get('size')
                        break
            if count is None and not df_points.empty and 'cluster_id' in df_points.columns:
                count = int((df_points['cluster_id'] == cid).sum())

            cols = st.columns(3)
            cols[0].metric("Количество точек", count if count is not None else 0)
            # Простейшие распределения признаков по имеющимся данным (если points есть)
            if not df_points.empty:
                sel = df_points[df_points['cluster_id'] == cid]
                if 'x' in sel.columns and 'y' in sel.columns:
                    cols[1].metric("X median", f"{sel['x'].median():.3f}")
                    cols[2].metric("Y median", f"{sel['y'].median():.3f}")
            else:
                st.caption("Распределения признаков будут доступны после расчёта проекций.")

            # 3) Действия / Экспорт
            st.divider()
            with st.container(border=True):
                st.subheader("Действия")
                if not df_points.empty and 'cluster_id' in df_points.columns:
                    csv_bytes = _export_cluster_csv(df_points, cid)
                    st.download_button(
                        "Экспорт кластера в CSV",
                        data=csv_bytes,
                        file_name=f"cluster_{cid}.csv",
                        mime="text/csv",
                        type="primary",
                    )
                else:
                    st.caption("Экспорт CSV недоступен: нет точек кластера.")

