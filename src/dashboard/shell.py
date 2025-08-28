"""Streamlit UI shell с боковым меню и корпоративным стилем"""
from __future__ import annotations

import os
import streamlit as st

try:
    from streamlit_option_menu import option_menu  # type: ignore
except Exception:  # пакет может отсутствовать в dev-образе
    option_menu = None  # type: ignore

# Тумблер: можно отключить расширенное меню без пересборки образа
if os.getenv("DASHBOARD_SIMPLE_MENU") in {"1", "true", "True", "yes"}:
    option_menu = None  # type: ignore


def _inject_corporate_style() -> None:
    """Вставляет CSS: фирменные цвета и базовая стилизация."""
    st.markdown(
        """
        <style>
        :root {
            --primary: #003057;
            --blue-500: #0095D9;
            --blue-300: #5DC9F5;
            --text: #0B1F35;
            --bg: #FFFFFF;
            --bg-2: #F5F8FF;
        }
        h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
        h1 { border-left: 6px solid var(--primary); padding-left: 10px; }
        .stButton>button { background: var(--primary) !important; color: #fff !important; border: 0; border-radius: 6px; }
        .stApp { background: var(--bg) !important; }
        .stSidebar { background: var(--bg-2) !important; }
        a { color: var(--blue-500); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Умный мониторинг",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_corporate_style()

    # Единый заголовок приложения (шапка)
    st.markdown("## 🧠 Умный мониторинг")

    # Боковое меню
    with st.sidebar:
        st.markdown("### ⚙️ Диагностическая система")
        st.caption("Сайдбар включён для демо. В продуктиве на главной странице будет скрыт.")
        # Кнопка формирования отчёта (пока заглушка)
        if st.button("📄 Сформировать отчёт"):
            st.toast("Извините, функционал находится в разработке", icon="⚠️")
        pages = [
            "Главная",
            "Загрузка исходных данных (для разработки)",
            "Признаки и группы (для анализа модели ИИ)",
            "Аномалии и прогноз",
            "Состояние оборудования",
        ]
        if option_menu is not None:
            choice = option_menu(
                "Навигация",
                pages,
                icons=["house", "cloud-upload", "diagram-3", "exclamation-triangle", "activity"],
                menu_icon="menu-button-wide",
                default_index=0,
            )
        else:
            st.info("Упрощённое меню без streamlit_option_menu", icon="ℹ️")
            choice = st.radio("Навигация", pages, index=0)

    # Программная навигация из страниц (через session_state.navigate_to)
    nav_target = st.session_state.pop("navigate_to", None) if isinstance(st.session_state, dict) else None
    if nav_target in pages:
        choice = nav_target

    # Маршрутизация
    if choice == "Главная":
        from dashboard.pages_shell import home as page
    elif choice == "Загрузка исходных данных (для разработки)":
        from dashboard.pages_shell import raw_upload as page
    elif choice == "Признаки и группы (для анализа модели ИИ)":
        from dashboard.pages_shell import features_clusters as page
    elif choice == "Аномалии и прогноз":
        from dashboard.pages_shell import anomalies_forecast as page
    else:
        from dashboard.pages_shell import system_monitoring as page

    page.render()


if __name__ == "__main__":
    main()
