"""Страница: System Monitoring

Функциональность:
- Встроенная панель Grafana (iframe), также ссылка на отдельное окно.
- Базовые метрики: количество загруженных сигналов, число кластеров, процент аномалий.
- Автообновление каждые 30 секунд.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
import requests


def _api() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000")


def _auth_headers() -> Dict[str, str]:
    token = st.session_state.get("token") if isinstance(st.session_state, dict) else None
    return {"Authorization": f"Bearer {token}"} if token else {}


@st.cache_data(show_spinner=False, ttl=20)
def fetch_equipment() -> List[Dict[str, Any]]:
    url = f"{_api()}/api/v1/equipment"
    try:
        r = requests.get(url, headers=_auth_headers(), timeout=20)
        if r.status_code == 200:
            return r.json() or []
    except Exception:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=20)
def fetch_signals_total(equipment_id: str) -> int:
    url = f"{_api()}/api/v1/signals"
    try:
        r = requests.get(url, params={"equipment_id": equipment_id, "page": 1, "page_size": 1}, headers=_auth_headers(), timeout=20)
        if r.status_code == 200:
            return int((r.json() or {}).get('total_count', 0))
    except Exception:
        pass
    return 0


@st.cache_data(show_spinner=False, ttl=20)
def fetch_anomalies_total_last_days(equipment_id: str, days: int = 30) -> int:
    end = datetime.utcnow().isoformat()
    start = (datetime.utcnow() - timedelta(days=days)).isoformat()
    url = f"{_api()}/api/v1/anomalies/{equipment_id}"
    try:
        r = requests.get(url, params={"start_date": start, "end_date": end, "page": 1, "page_size": 1}, headers=_auth_headers(), timeout=30)
        if r.status_code == 200:
            return int((r.json() or {}).get('total_anomalies', 0))
    except Exception:
        pass
    return 0


@st.cache_data(show_spinner=False, ttl=20)
def fetch_cluster_count() -> int:
    # Предпочитаем centroids в distribution; иначе считаем уникальные cluster_id из points; fallback — по labels
    try:
        r = requests.get(f"{_api()}/admin/clustering/distribution", headers=_auth_headers(), timeout=30)
        if r.status_code == 200:
            data = r.json() or {}
            cents = data.get('centroids')
            if isinstance(cents, list) and len(cents) > 0:
                return len(cents)
            pts = data.get('points')
            if isinstance(pts, list):
                uniq = {p.get('cluster_id') for p in pts if 'cluster_id' in p}
                return len(uniq)
    except Exception:
        pass
    try:
        r = requests.get(f"{_api()}/admin/clustering/labels", headers=_auth_headers(), timeout=20)
        if r.status_code == 200:
            labels = r.json() or []
            uniq = {row.get('cluster_id') for row in labels}
            return len(uniq)
    except Exception:
        pass
    return 0


def _auto_refresh(seconds: int = 30) -> None:
    # Вставляем meta-refresh только на этой странице
    st.markdown(f"<meta http-equiv='refresh' content='{int(seconds)}'>", unsafe_allow_html=True)


def _grafana_url() -> str:
    base = os.getenv("GRAFANA_URL", "http://localhost:3000")
    # Можно переопределить конкретную доску через переменную окружения; иначе дефолтный путь
    path = os.getenv("GRAFANA_DASHBOARD_PATH", "/d/diag/overview")
    params = os.getenv("GRAFANA_DASHBOARD_PARAMS", "?orgId=1&refresh=30s")
    # Если уже полный URL — возвращаем как есть
    if base.startswith("http://") or base.startswith("https://"):
        if base.endswith('/') and path.startswith('/'):
            base = base[:-1]
        return f"{base}{path}{params}"
    return "http://localhost:3000/d/diag/overview?orgId=1&refresh=30s"


def render() -> None:
    st.title("🖥️ System Monitoring")
    st.caption("Системные метрики и панель Grafana")

    # Автообновление каждые 30 секунд
    _auto_refresh(30)

    # 1) Фильтры / Выбор (опционально)
    with st.container(border=True):
        st.subheader("Фильтры")
        st.caption("В текущей версии метрики агрегируются по всему доступному оборудованию")

    # 2) Визуализация
    with st.container(border=True):
        st.subheader("Базовые метрики проекта")
        eq = fetch_equipment()
        if not eq:
            st.info("Оборудование не найдено или нет доступа к API")
        total_signals = 0
        total_anomalies = 0
        if eq:
            for e in eq:
                eid = str(e.get('id'))
                total_signals += fetch_signals_total(eid)
                total_anomalies += fetch_anomalies_total_last_days(eid, days=30)
        cluster_count = fetch_cluster_count()
        anomaly_pct = round((total_anomalies / max(1, total_signals)) * 100, 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Загруженных сигналов", total_signals)
        c2.metric("Кластеров", cluster_count)
        c3.metric("Процент аномалий (30 дней)", f"{anomaly_pct}%", help=f"Всего аномалий: {total_anomalies}")

    with st.container(border=True):
        st.subheader("Grafana")
        url = _grafana_url()
        st.write("Встроенная панель ниже. При проблемах с X-Frame-Options используйте ссылку для открытия в новой вкладке.")
        st.link_button("Открыть в Grafana", url)
        try:
            from streamlit import components
            components.v1.iframe(src=url, height=800, scrolling=True)
        except Exception:
            st.write("Невозможно отобразить iframe. Откройте панель по ссылке выше.")

    # 3) Действия / Экспорт (резерв)
    st.divider()
    with st.container(border=True):
        st.subheader("Действия")
        st.caption("Здесь можно добавить экспорт/кнопки управления мониторингом при необходимости")

