"""Домашняя страница: список двигателей (лёгкий демо‑режим и безопасные плейсхолдеры картинок)."""
from __future__ import annotations

import os
from io import BytesIO
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont
import requests
import streamlit as st


def _get_api_base() -> str:
    # В compose у нас API_URL, но поддержим и API_BASE_URL для совместимости
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def _demo_mode() -> bool:
    # Включаем лёгкий демо‑режим, чтобы не зависеть от API и внешних картинок
    return (os.getenv("DASHBOARD_TEST_DATA", "1").lower() in {"1", "true", "yes", "on"})


def _placeholder_image(status: str = "ok", size: tuple[int, int] = (160, 90)) -> Image.Image:
    """Генерирует простой локальный плейсхолдер без внешних ресурсов.

    Цвет фона зависит от статуса: ok — зелёный, warn — жёлтый, crit — красный.
    """
    color_map = {
        "ok": (33, 186, 69),       # зелёный
        "warn": (242, 192, 55),    # жёлтый
        "crit": (219, 40, 40),     # красный
    }
    # Прозрачная подложка (RGBA), только рамка и текст
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # тонкая полупрозрачная рамка
    draw.rectangle([(0, 0), (size[0]-1, size[1]-1)], outline=(0, 0, 0, 96), width=2)
    # текст по центру
    text = {
        "ok": "Двигатель: Хорошо",
        "warn": "Двигатель: Внимание",
        "crit": "Двигатель: Критично",
    }.get(status.lower(), "Двигатель")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore
    # Вычисляем размер текста кросс-версийно: сначала пробуем textbbox, иначе — fallback
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        # Fallback: оцениваем ширину текстовой строки без точной метрики
        try:
            # Pillow>=8: у ImageFont есть getlength; примем высоту шрифта приблизительно
            w = int(font.getlength(text)) if hasattr(font, "getlength") else max(8 * len(text), 40)  # type: ignore[union-attr]
            h = getattr(font, "size", 14) or 14  # type: ignore[union-attr]
        except Exception:
            w, h = max(8 * len(text), 40), 14
    x = (size[0] - w) // 2
    y = (size[1] - h) // 2
    draw.text((x, y), text, fill=(0, 0, 0, 230), font=font)
    return img


def _load_engine_image(size: tuple[int, int] = (160, 90)) -> Image.Image | None:
    """Пытаемся загрузить пользовательскую картинку двигателя.

    Источники (по порядку):
    1) Переменная окружения DASHBOARD_ENGINE_IMAGE (абсолютный путь).
    2) Встроенный путь в репозитории: src/dashboard/assets/engine.png
    Возвращаем PIL.Image или None, если не нашли.
    """
    candidates: list[str] = []
    # Сначала пробуем встроенный ассет в репозитории (постоянно доступен)
    here = os.path.dirname(__file__)
    assets_path = os.path.normpath(os.path.join(here, "..", "assets", "engine.png"))
    candidates.append(assets_path)
    # Затем необязательная переменная окружения (fallback)
    env_path = os.getenv("DASHBOARD_ENGINE_IMAGE")
    if env_path:
        candidates.append(env_path)
    for p in candidates:
        try:
            if p and os.path.isfile(p):
                src = Image.open(p).convert("RGBA")
                # Сохраняем пропорции: вписываем в размер и центрируем на прозрачном фоне
                img = Image.new("RGBA", size, (0, 0, 0, 0))
                tmp = src.copy()
                tmp.thumbnail(size, Image.LANCZOS)
                ox = (size[0] - tmp.width) // 2
                oy = (size[1] - tmp.height) // 2
                img.paste(tmp, (ox, oy), mask=tmp)
                return img
        except Exception:
            continue
    return None


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

    engines: list[Dict[str, Any]] = []
    try:
        # Пытаемся получить список оборудования; если эндпоинт отличается — позже адаптируем
        if not _demo_mode():
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

    # Если список есть — сетка карточек двигателей с безопасными локальными плейсхолдерами
    st.subheader("Двигатели")
    # Сделаем блоки компактнее: больше столбцов, меньше изображение
    grid_cols = 4
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
                    status = (item.get("status") or "ok").lower()
                    # Картинка: сначала пробуем пользовательскую, иначе плейсхолдер
                    img = _load_engine_image() or _placeholder_image(status=status)
                    st.image(img, caption=str(title), width="content")

                    # Чуть компактнее описание и метрики
                    if item.get("description"):
                        st.caption(item.get("description"))
                    cols2 = st.columns(2)
                    with cols2[0]:
                        st.metric("Идентификатор", str(item.get("id")))
                    with cols2[1]:
                        # Светофор статуса
                        if status not in {"ok", "warn", "crit"}:
                            status = "ok"
                        color = {"ok": "#21ba45", "warn": "#f2c037", "crit": "#db2828"}[status]
                        label = {"ok": "Все хорошо", "warn": "Требует внимания", "crit": "Критично"}[status]
                        st.markdown(f"<div style='display:flex;align-items:center;gap:6px'>"
                                    f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color}'></span>"
                                    f"<span>{label}</span>"
                                    f"</div>", unsafe_allow_html=True)
                    if st.button("Подробнее", key=f"open_engine_{item.get('id')}"):
                        # Сохраняем выбранный двигатель и переходим на первую рабочую страницу
                        st.session_state["selected_equipment_id"] = item.get("id")
                        st.session_state["selected_equipment_name"] = title
                        st.session_state["selected_equipment_status"] = status
                        st.session_state["came_from_home"] = True
                        st.session_state["navigate_to"] = "Аномалии и прогноз"
                        st.rerun()
