"""Домашняя страница: список двигателей (заглушка при отсутствии данных)."""
from __future__ import annotations

import os
import requests
import streamlit as st


def _get_api_base() -> str:
    # В compose у нас API_URL, но поддержим и API_BASE_URL для совместимости
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def render() -> None:
    st.title("🏠 Главная")
    st.caption("Обзор оборудования и быстрые ссылки")

    # Уведомление о демонстрационном сайдбаре
    st.info(
        "Сайдбар отображается только в демонстрационном варианте для удобства навигации. В продуктиве на главной странице он будет скрыт.",
        icon="ℹ️",
    )

    # Кнопка настроек уведомлений (заглушка)
    if st.button("🔔 Настройка уведомлений"):
        st.toast("Извините, функционал находится в разработке", icon="⚠️")

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
        engines = []

    if not engines:
        st.info(
            "Тут будет список двигателей. Данный функционал ещё в разработке.",
            icon="ℹ️",
        )
        st.write("\n")
        st.write("Пока вы можете перейти на другие разделы через меню слева.")
        # Демо-оборудование, если API недоступен/данных нет
        engines = [
            {"id": "demo-ok", "name": "Двигатель A (демо)", "status": "ok"},
            {"id": "demo-warn", "name": "Двигатель B (демо)", "status": "warn"},
            {"id": "demo-crit", "name": "Двигатель C (демо)", "status": "crit"},
            {"id": "demo-ok-2", "name": "Двигатель D (демо)", "status": "ok"},
            {"id": "demo-warn-2", "name": "Двигатель E (демо)", "status": "warn"},
            {"id": "demo-ok-3", "name": "Двигатель F (демо)", "status": "ok"},
        ]
        # Падать вниз на общий рендер карточек

    # Если список есть — сетка карточек двигателей с изображениями-заглушками
    st.subheader("Двигатели")
    grid_cols = 3
    rows = (len(engines) + grid_cols - 1) // grid_cols
    idx = 0
    for _ in range(rows):
        cols = st.columns(grid_cols)
        for c in cols:
            if idx >= len(engines):
                break
            item = engines[idx]
            idx += 1
            with c:
                with st.container(border=True):
                    title = item.get("name") or item.get("id") or "Двигатель"
                    # Источник картинки: приоритет локального файла, затем image_url из API, затем placeholder
                    local_img = "E:/двигатель.png"
                    img_src = local_img if os.path.exists(local_img) else (item.get("image_url") or "https://via.placeholder.com/320x180.png?text=Engine")
                    st.image(img_src, caption=str(title), use_column_width=True)
                    st.caption(item.get("description") or "Без описания")
                    cols2 = st.columns(3)
                    with cols2[0]:
                        st.metric("ID", str(item.get("id")))
                    with cols2[1]:
                        # Светофор статуса
                        status = (item.get("status") or "ok").lower()
                        if status not in {"ok", "warn", "crit"}:
                            status = "ok"
                        color = {"ok": "#21ba45", "warn": "#f2c037", "crit": "#db2828"}[status]
                        label = {"ok": "Все хорошо", "warn": "Требует внимания", "crit": "Критично"}[status]
                        st.markdown(f"<div style='display:flex;align-items:center;gap:6px'>"
                                    f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color}'></span>"
                                    f"<span>{label}</span>"
                                    f"</div>", unsafe_allow_html=True)
                    with cols2[2]:
                        st.metric("Обновление", item.get("updated_at") or "—")
                    if st.button("Подробнее", key=f"open_engine_{item.get('id')}"):
                        # Сохраняем выбранный двигатель и переходим на первую рабочую страницу
                        st.session_state["selected_equipment_id"] = item.get("id")
                        st.session_state["selected_equipment_name"] = title
                        st.session_state["selected_equipment_status"] = status
                        st.session_state["came_from_home"] = True
                        st.session_state["navigate_to"] = "Аномалии и прогноз"
                        st.rerun()
