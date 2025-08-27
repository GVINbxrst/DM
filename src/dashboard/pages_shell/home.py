"""Домашняя страница: список двигателей (заглушка при отсутствии данных)."""
from __future__ import annotations

import os
import requests
import streamlit as st


def _get_api_base() -> str:
    # В compose у нас API_URL, но поддержим и API_BASE_URL для совместимости
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def render() -> None:
    st.title("🏠 Домашняя")
    st.caption("Обзор оборудования и быстрые ссылки")

    api = _get_api_base()
    headers = {}
    token = st.session_state.get("token") if isinstance(st.session_state, dict) else None
    if token:
        headers["Authorization"] = f"Bearer {token}"

    engines = []
    try:
        # Пытаемся получить список оборудования; если эндпоинт отличается — позже адаптируем
        r = requests.get(f"{api}/api/v1/equipment", headers=headers, timeout=10)
        if r.status_code == 200:
            engines = r.json() or []
    except Exception:
        pass

    if not engines:
        st.info(
            "Тут будет список двигателей. Данный функционал ещё в разработке.",
            icon="ℹ️",
        )
        st.write("\n")
        st.write("Пока вы можете перейти на другие разделы через меню слева.")
        return

    # Если список есть — простой рендер карточек
    for item in engines:
        with st.container(border=True):
            title = item.get("name") or item.get("id") or "Двигатель"
            st.subheader(str(title))
            st.caption(item.get("description") or "Без описания")
            cols = st.columns(3)
            with cols[0]:
                st.metric("ID", str(item.get("id")))
            with cols[1]:
                st.metric("Статус", item.get("status") or "n/a")
            with cols[2]:
                st.metric("Последнее обновление", item.get("updated_at") or "—")
            st.button("Открыть тренды", key=f"open_trends_{item.get('id')}")
