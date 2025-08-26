"""Streamlit UI shell с боковым меню и корпоративным стилем"""
from __future__ import annotations

import os
import streamlit as st
from streamlit_option_menu import option_menu


def _inject_corporate_style() -> None:
    """Вставляет CSS со стилями Газпром нефти: синие оттенки, тёмно-синий акцент."""
    st.markdown(
        """
        <style>
                :root {
                    --primary: #003057; /* тёмно-синий */
                    --blue-500: #0095D9; /* фирменный голубой */
                    --blue-300: #5DC9F5;
          --text: #0B1F35;
          --bg: #FFFFFF;
          --bg-2: #F5F8FF;
        }
        /* Заголовки */
        h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
        h1 { border-left: 6px solid var(--primary); padding-left: 10px; }
        /* Кнопки */
        .stButton>button {
          background: var(--primary) !important;
          color: #fff !important;
          border: 0; border-radius: 6px;
        }
        /* Блоки */
        .stApp { background: var(--bg) !important; }
        .stSidebar { background: var(--bg-2) !important; }
        a { color: var(--blue-500); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # Базовая конфигурация страницы
    st.set_page_config(
    page_title="DiagMod — Диагностика",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_corporate_style()

    # Боковое меню
    with st.sidebar:
        st.markdown("### ⚙️ Диагностическая система")
        choice = option_menu(
            "Навигация",
            [
                "Сырые сигналы и загрузка",
                "Признаки и кластеры",
                "Аномалии и прогноз",
                "Системный мониторинг",
            ],
            icons=[
                "cloud-upload",
                "diagram-3",
                "exclamation-triangle",
                "activity",
            ],
            menu_icon="menu-button-wide",
            default_index=0,
        )

    # Маршрутизация на страницы
    if choice == "Сырые сигналы и загрузка":
        from src.dashboard.pages_shell import raw_upload as page
    elif choice == "Признаки и кластеры":
        from src.dashboard.pages_shell import features_clusters as page
    elif choice == "Аномалии и прогноз":
        from src.dashboard.pages_shell import anomalies_forecast as page
    else:
        from src.dashboard.pages_shell import system_monitoring as page

    page.render()


if __name__ == "__main__":
    main()
