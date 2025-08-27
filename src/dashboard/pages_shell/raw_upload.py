"""Страница: Raw Signals & Upload

Требования:
- Виджет загрузки CSV.
- Формат: заголовок current_R,current_S,current_T; строки — три значения или пусто.
- Парсинг и предпросмотр (первые 20 строк).
- График фаз с переключателями.
- Сохранение через API /api/v1/upload (или напрямую позже).
"""
from __future__ import annotations

import io
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests


EXPECTED_HEADER = ["current_R", "current_S", "current_T"]


def _get_api_base() -> str:
    return os.getenv("API_URL") or os.getenv("API_BASE_URL") or "http://api:8000"


def _decode_csv(file_bytes: bytes) -> str:
    for enc in ("utf-8", "cp1251", "utf-8-sig"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    # как fallback пытаемся как latin-1
    return file_bytes.decode("latin-1", errors="ignore")


def _parse_csv(text: str) -> pd.DataFrame:
    # Пытаемся стандартно
    try:
        df = pd.read_csv(io.StringIO(text), sep=",", engine="python")
    except Exception:
        # жесткий разбор строк, если CSV «одной колонкой»
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame(columns=EXPECTED_HEADER)
        header = lines[0].replace(" ", "")
        cols = header.split(",")
        data_rows = [ln.split(",") for ln in lines[1:]]
        df = pd.DataFrame(data_rows, columns=cols)

    # Нормализуем набор колонок
    cols_lower = [c.strip() for c in df.columns]
    rename_map = {}
    for c in cols_lower:
        c_norm = c.replace(" ", "")
        if c_norm.lower() in {"current_r", "r", "phase_r"}:
            rename_map[c] = "current_R"
        elif c_norm.lower() in {"current_s", "s", "phase_s"}:
            rename_map[c] = "current_S"
        elif c_norm.lower() in {"current_t", "t", "phase_t"}:
            rename_map[c] = "current_T"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in EXPECTED_HEADER:
        if col not in df.columns:
            df[col] = None

    # Оставим только нужные колонки и приведём к числу
    df = df[EXPECTED_HEADER]
    for col in EXPECTED_HEADER:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Уберём строки, где все три NaN
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    return df


def _preview_table(df: pd.DataFrame) -> None:
    st.subheader("Предпросмотр (первые 20 строк)")
    st.dataframe(df.head(20), use_container_width=True)


def _plot_phases(df: pd.DataFrame, default: List[str] | None = None) -> None:
    st.subheader("График фаз тока")
    phases = [c for c in EXPECTED_HEADER if c in df.columns]
    if not phases:
        st.info("Нет колонок для графика")
        return
    sel = st.multiselect("Фазы", phases, default=default or phases)
    if not sel:
        st.info("Выберите хотя бы одну фазу")
        return
    fig = go.Figure()
    x = list(range(len(df)))
    # фирменные оттенки: тёмно-синий и голубой, плюс нейтральный серо-синий
    colors = {
        "current_R": "#003057",
        "current_S": "#0095D9",
        "current_T": "#5DC9F5",
    }
    for col in sel:
        fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col, line=dict(color=colors.get(col))))
    fig.update_layout(height=400, xaxis_title="Отсчёт", yaxis_title="Ток (А)", plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)


def _to_canonical_csv_bytes(df: pd.DataFrame) -> bytes:
    # Пересохраняем с каноническим заголовком для /api/v1/upload
    out = io.StringIO()
    df_out = df[EXPECTED_HEADER].copy()
    df_out.to_csv(out, index=False)
    text = out.getvalue()
    # Убеждаемся, что первая строка совпадает точно (без пробелов)
    lines = text.splitlines()
    if not lines:
        return b""
    lines[0] = ",".join(EXPECTED_HEADER)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _upload_via_api(csv_bytes: bytes, equipment_id: str | None, sample_rate: int | None, description: str | None) -> Tuple[bool, str]:
    api = _get_api_base()
    url = f"{api}/api/v1/upload"
    files = {
        "file": ("signal.csv", csv_bytes, "text/csv"),
    }
    data = {}
    if equipment_id:
        data["equipment_id"] = equipment_id
    if sample_rate:
        data["sample_rate"] = str(sample_rate)
    if description:
        data["description"] = description

    headers = {}
    token = st.session_state.get("token") if isinstance(st.session_state, dict) else None
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.post(url, files=files, data=data, headers=headers, timeout=120)
        if r.status_code == 200:
            return True, r.text
        return False, f"{r.status_code}: {r.text}"
    except Exception as e:
        return False, f"request_error: {e}"


def render() -> None:
    st.title("📥 Загрузка исходных данных (для разработки)")
    st.caption("CSV формата current_R,current_S,current_T — предпросмотр, графики и отправка в API")

    # 1) Фильтры / Выбор
    with st.container(border=True):
        st.subheader("Выбор и параметры загрузки")
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader("Выберите CSV-файл", type=["csv"], accept_multiple_files=False)
        with col2:
            sample_rate = st.number_input("Частота дискретизации, Гц", min_value=1000, max_value=100000, value=25600, step=100)
        equipment_id = st.text_input("Идентификатор оборудования (UUID, необязательно)")
        description = st.text_input("Описание (опционально)")

    if uploaded is None:
        st.info("Загрузите файл CSV для предпросмотра и отправки в БД")
        return

    # Кэшируем содержимое файла в сессии, чтобы не терять при повторных нажатиях
    if "uploaded_bytes" not in st.session_state:
        st.session_state.uploaded_bytes = uploaded.getvalue()
    elif uploaded.size != len(st.session_state.uploaded_bytes):
        st.session_state.uploaded_bytes = uploaded.getvalue()

    file_bytes = st.session_state.uploaded_bytes
    csv_text = _decode_csv(file_bytes)
    df = _parse_csv(csv_text)

    if df.empty:
        st.error("Не удалось распарсить CSV или файл пустой")
        return

    # 2) Визуализация
    with st.container(border=True):
        st.subheader("Визуализация")
        _preview_table(df)
        _plot_phases(df)

    # 3) Действия / Экспорт
    st.divider()
    with st.container(border=True):
        st.subheader("Действия")
        st.write("Файл будет отправлен на API /api/v1/upload. Заголовок будет нормализован к 'current_R,current_S,current_T'.")
        if st.button("Сохранить через API", type="primary"):
            with st.spinner("Отправка на сервер..."):
                csv_bytes = _to_canonical_csv_bytes(df)
                ok, msg = _upload_via_api(csv_bytes, equipment_id.strip() or None, int(sample_rate) if sample_rate else None, description.strip() or None)
                if ok:
                    st.success("Файл успешно сохранён")
                    try:
                        st.json(pd.read_json(io.StringIO(msg)) if msg.strip().startswith("{") else msg)
                    except Exception:
                        st.text(msg)
                else:
                    st.error(f"Ошибка сохранения: {msg}")
