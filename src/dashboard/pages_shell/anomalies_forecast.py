"""Страница: Anomalies & Forecast

Функциональность:
- График аномалий (confidence/score) по времени, цвет по модели.
- Прогноз тренда RMS (с доверительным интервалом) из /api/v1/forecast_rms.
- Фильтр по оборудованию и периоду времени.
- Легенда «тип дефекта» (cluster_labels) при наличии.
- Карточки статуса: нормальные сигналы (оценка), аномалии, топ-3 дефекта.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests
from dashboard.utils_report import build_demo_pdf_report


def _api() -> str:
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def _auth_headers() -> Dict[str, str]:
    token = st.session_state.get("token") if isinstance(st.session_state, dict) else None
    return {"Authorization": f"Bearer {token}"} if token else {}


def _demo_mode() -> bool:
    # По умолчанию включаем демо-режим (можно отключить переменной окружения DASHBOARD_TEST_DATA=0)
    return (os.getenv("DASHBOARD_TEST_DATA", "1").lower() in {"1", "true", "yes", "on"})


@st.cache_data(show_spinner=False, ttl=120)
def fetch_equipment() -> List[Dict[str, Any]]:
    if _demo_mode():
        return []
    url = f"{_api()}/api/v1/equipment"
    try:
        r = requests.get(url, headers=_auth_headers(), timeout=15)
        if r.status_code == 200:
            return r.json() or []
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=60)
def fetch_anomalies(equipment_id: str, start: Optional[str], end: Optional[str], min_conf: Optional[float]) -> Dict[str, Any] | None:
    url = f"{_api()}/api/v1/anomalies/{equipment_id}"
    params: Dict[str, Any] = {}
    if start:
        params['start_date'] = start
    if end:
        params['end_date'] = end
    if min_conf is not None:
        params['min_confidence'] = min_conf
    try:
        r = requests.get(url, params=params, headers=_auth_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Ошибка загрузки аномалий: {e}")
    return None


@st.cache_data(show_spinner=False, ttl=60)
def fetch_forecast(equipment_id: str, steps: int) -> Dict[str, Any] | None:
    url = f"{_api()}/api/v1/forecast_rms/{equipment_id}"
    try:
        r = requests.get(url, params={"steps": steps}, headers=_auth_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
        # fallback на legacy
        r = requests.get(f"{_api()}/api/v1/anomalies/forecast_rms/{equipment_id}", params={"steps": steps}, headers=_auth_headers(), timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Ошибка загрузки прогноза: {e}")
    return None


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


@st.cache_data(show_spinner=False, ttl=60)
def fetch_signals_total(equipment_id: str) -> int:
    # Используем total_count как оценку общего количества сигналов по оборудованию
    url = f"{_api()}/api/v1/signals"
    try:
        r = requests.get(url, params={"equipment_id": equipment_id, "page": 1, "page_size": 1}, headers=_auth_headers(), timeout=30)
        if r.status_code == 200:
            return int((r.json() or {}).get('total_count', 0))
    except Exception as e:
        st.warning(f"Ошибка получения общего числа сигналов: {e}")
    return 0


def _map_feature_to_defect(dist: Dict[str, Any] | None, labels: List[Dict[str, Any]]) -> Dict[str, str]:
    # feature_id (str) -> defect_id
    if not dist:
        return {}
    # Строим index: feature_id -> cluster_id
    feature2cluster: Dict[str, int] = {}
    points = dist.get('points') if isinstance(dist, dict) else None
    if isinstance(points, list):
        for p in points:
            fid = str(p.get('feature_id')) if p.get('feature_id') is not None else None
            cid = p.get('cluster_id')
            if fid and cid is not None:
                try:
                    feature2cluster[fid] = int(cid)
                except Exception:
                    continue
    # labels: cluster_id -> defect_id
    clabels: Dict[int, str] = {}
    for row in labels:
        try:
            clabels[int(row.get('cluster_id'))] = str(row.get('defect_id') or row.get('description') or '')
        except Exception:
            continue
    # map features to defect
    out: Dict[str, str] = {}
    for fid, cid in feature2cluster.items():
        if cid in clabels:
            out[fid] = clabels[cid]
    return out


def _anomalies_df(data: Dict[str, Any] | None) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    arr = data.get('anomalies') or []
    if not isinstance(arr, list):
        return pd.DataFrame()
    rows = []
    for it in arr:
        rows.append({
            'id': it.get('id'),
            'feature_id': it.get('feature_id'),
            'ts': it.get('detected_at'),
            'confidence': it.get('confidence'),
            'severity': it.get('severity'),
            'model': it.get('model_name') or 'model',
        })
    df = pd.DataFrame(rows)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    return df


def _plot_anomalies(df: pd.DataFrame) -> None:
    st.subheader("Аномалии по времени")
    if df.empty:
        st.info("Аномалии не найдены в выбранном периоде")
        return
    fig = px.scatter(df.sort_values('ts'), x='ts', y='confidence', color='model', hover_data=['feature_id', 'severity'])
    fig.update_traces(mode='lines+markers')
    fig.update_layout(height=380, yaxis_title='Уверенность', xaxis_title='Время')
    st.plotly_chart(fig, width="stretch")


def _plot_forecast(fc: Dict[str, Any] | None) -> None:
    st.subheader("Прогноз тренда (RMS)")
    if not fc:
        st.info("Прогноз недоступен. Попробуйте увеличить горизонт или запустить генерацию.")
        return
    forecast = fc.get('forecast') or []
    if not forecast:
        st.info("Нет данных прогноза")
        return
    df = pd.DataFrame(forecast)
    # Ожидается timestamp + rms; lower_ci/upper_ci могут прийти отдельными массивами той же длины
    lower = fc.get('lower_ci')
    upper = fc.get('upper_ci')
    if isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(df):
        df['lower_ci'] = lower
        df['upper_ci'] = upper
    # Рисуем
    fig = go.Figure()
    if 'timestamp' in df.columns:
        x = pd.to_datetime(df['timestamp'])
    else:
        x = list(range(len(df)))
    # Выбираем числовую метрику для прогноза
    y_col = None
    for cand in ['rms', 'value', 'y']:
        if cand in df.columns:
            y_col = cand
            break
    if y_col is None:
        # берем первую числовую колонку, исключая служебные
        exclude = {'timestamp', 'lower_ci', 'upper_ci'}
        num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            y_col = num_cols[0]
        elif len(df.columns) > 1:
            y_col = df.columns[1]
        else:
            y_col = df.columns[0]
    fig.add_trace(go.Scatter(x=x, y=df[y_col], mode='lines', name='Прогноз'))
    if 'lower_ci' in df.columns and 'upper_ci' in df.columns:
        fig.add_traces([
            go.Scatter(x=x, y=df['upper_ci'], mode='lines', line=dict(width=0), showlegend=False),
            go.Scatter(x=x, y=df['lower_ci'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31,119,180,0.2)', name='95% ДИ')
        ])
    thr = fc.get('threshold')
    if thr is not None:
        fig.add_hline(y=thr, line_dash='dash', line_color='red', annotation_text='порог')
    fig.update_layout(height=380, yaxis_title='RMS', xaxis_title='Время')
    st.plotly_chart(fig, width="stretch")


def _demo_anomalies(status: str, start: datetime, end: datetime, min_conf: float) -> pd.DataFrame:
    # Генерируем точечные аномалии с разной интенсивностью
    import math
    # Малый фиксированный объём данных для наглядности
    n = {"ok": 6, "warn": 10, "crit": 14}.get(status, 8)
    rows = []
    for i in range(n):
        t = start + timedelta(hours=i)
        base = 0.25 if status == "ok" else (0.55 if status == "warn" else 0.8)
        jitter = 0.1 * math.sin(i * 1.7)
        conf = max(0.01, min(0.99, base + jitter))
        if conf < min_conf:
            continue
        rows.append({
            'id': f'demo-{i}',
            'feature_id': f'feat-{100+i}',
            'ts': t,
            'confidence': conf,
            'severity': 'high' if conf > 0.75 else ('medium' if conf > 0.5 else 'low'),
            'model': 'demo-model'
        })
    df = pd.DataFrame(rows)
    return df


def _demo_forecast(status: str, steps: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    steps = min(24, max(6, int(steps)))
    base = 0.4 if status == "ok" else (0.55 if status == "warn" else 0.8)
    trend = 0.002 if status == "ok" else (0.01 if status == "warn" else 0.03)
    fc = []
    low, up = [], []
    for i in range(steps):
        v = base + trend * i
        fc.append({'timestamp': (now + timedelta(hours=i)).isoformat(), 'rms': v})
        low.append(max(0.0, v - 0.05))
        up.append(v + 0.05)
    return {'forecast': fc, 'lower_ci': low, 'upper_ci': up, 'threshold': 0.9 if status == 'crit' else (0.75 if status == 'warn' else 0.95)}


def render() -> None:
    st.title("⚠️ Аномалии и прогноз")
    st.caption("Детекция аномалий и прогноз тренда с доверительным интервалом")

    # 1) Фильтры / Выбор
    with st.container(border=True):
        st.subheader("Фильтры")
        if _demo_mode():
            eq = [
                {"id": "demo-ok", "name": "Двигатель A (демо)", "status": "ok"},
                {"id": "demo-warn", "name": "Двигатель B (демо)", "status": "warn"},
                {"id": "demo-crit", "name": "Двигатель C (демо)", "status": "crit"},
            ]
        else:
            eq = fetch_equipment()
            if not eq:
                eq = [
                    {"id": "demo-ok", "name": "Двигатель A (демо)", "status": "ok"},
                    {"id": "demo-warn", "name": "Двигатель B (демо)", "status": "warn"},
                    {"id": "demo-crit", "name": "Двигатель C (демо)", "status": "crit"},
                ]
        eq_options = {f"{e.get('name','')} ({e.get('id')})": e.get('id') for e in eq}
        colf = st.columns([2, 2, 1, 1])
        with colf[0]:
            eq_key = st.selectbox("Оборудование", list(eq_options.keys()))
        with colf[1]:
            period = st.select_slider("Период", options=[7, 14, 30, 60, 90], value=30, format_func=lambda d: f"{d} дней")
        with colf[2]:
            min_conf = st.slider("Мин. уверенность", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        with colf[3]:
            fc_steps = st.slider("Горизонт", min_value=12, max_value=72, value=24, step=6)

    # Определяем статус выбранного двигателя
    def _status_of(eid: str) -> str:
        for e in eq:
            if str(e.get('id')) == str(eid):
                return (e.get('status') or 'ok').lower()
        return (st.session_state.get('selected_equipment_status') or 'ok').lower() if isinstance(st.session_state, dict) else 'ok'

    equipment_id = eq_options.get(eq_key)
    end = datetime.utcnow()
    start = end - timedelta(days=int(period))
    # Для простых демо-графиков используем небольшие тестовые данные
    df_anom = _demo_anomalies(_status_of(str(equipment_id)), start, end, float(min_conf))

    # 2) Визуализация
    with st.container(border=True):
        st.subheader("Визуализация")
        st.caption("Демо: графики построены на небольших тестовых данных. Легенда отключена.")

        # Карточки статуса
        total_anom = int(len(df_anom))
        total_signals = max(30, total_anom * 3)
        normal_est = max(0, total_signals - total_anom)
        top_def: List[str] = []
        c1, c2, c3 = st.columns(3)
        c1.metric("Количество нормальных сигналов (оценка)", normal_est)
        c2.metric("Количество аномалий", total_anom)
        c3.metric("Топ-3 дефекта", ", ".join(map(str, top_def)) if top_def else "Нет данных")

        # График аномалий
        _plot_anomalies(df_anom)

        # Прогноз (только демо-данные)
        fc = _demo_forecast(_status_of(str(equipment_id)), int(fc_steps))
        _plot_forecast(fc)

        # Мини‑таблица последних демо‑сигналов
        demo_signals = pd.DataFrame([
            {"id": "rs-1", "equipment": eq_key, "status": _status_of(str(equipment_id)), "rms": 0.55, "ts": end.strftime('%Y-%m-%d %H:%M')},
            {"id": "rs-2", "equipment": eq_key, "status": _status_of(str(equipment_id)), "rms": 0.58, "ts": (end - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')},
        ])
        st.caption("Последние сигналы (демо)")
    st.dataframe(demo_signals, width="stretch")

    # 3) Действия / Экспорт
    st.divider()
    with st.container(border=True):
        st.subheader("Действия")
        # экспорт аномалий в CSV
        if not df_anom.empty:
            out = df_anom.copy()
            out['ts'] = out['ts'].astype(str)
            csv_bytes = out.to_csv(index=False).encode('utf-8')
            st.download_button("Экспорт аномалий (CSV)", data=csv_bytes, file_name="anomalies.csv", mime="text/csv", type="primary")

        # Кнопка сформировать отчёт, только если пришли с Главной и выбран двигатель
        if isinstance(st.session_state, dict) and st.session_state.get("came_from_home") and equipment_id:
            eq_name = st.session_state.get("selected_equipment_name") or str(equipment_id)
            demo_status = (st.session_state.get("selected_equipment_status") or "ok").lower()
            pdf_bytes = None
            if st.button("Сформировать отчёт"):
                try:
                    pdf_bytes = build_demo_pdf_report(str(eq_name), demo_status if demo_status in {"ok","warn","crit"} else "ok")
                except Exception as e:
                    st.error(f"Не удалось сформировать отчёт: {e}")
            if pdf_bytes:
                st.download_button(
                    label="Скачать отчёт (PDF)",
                    data=pdf_bytes,
                    file_name=f"report_{equipment_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                )

