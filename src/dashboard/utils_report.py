from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from fpdf import FPDF


def _demo_series(status: str) -> Dict[str, Tuple[list[float], list[float]]]:
    # Генерирует тестовые ряды под разные сценарии
    x = list(range(60))
    if status == "crit":
        y1 = [0.5 + i*0.03 for i in x]  # рост RMS
        y2 = [0.2 + (0.6 if 20 < i < 35 else 0) for i in x]  # всплеск вибраций
        y3 = [0.1 + (0.9 if i % 13 == 0 else 0) for i in x]  # импульсные аномалии
    elif status == "warn":
        y1 = [0.5 + i*0.005 for i in x]  # слабо возрастающий тренд
        y2 = [0.2 + (0.3 if 25 < i < 40 else 0) for i in x]
        y3 = [0.1 + (0.5 if i % 17 == 0 else 0) for i in x]
    else:  # ok
        y1 = [0.4 + 0.02*(i % 10) for i in x]
        y2 = [0.2 + 0.05*((i//7) % 3) for i in x]
        y3 = [0.1 for _ in x]
    return {
        "RMS": (x, y1),
        "Вибрации": (x, y2),
        "Аномалии": (x, y3),
    }


def _plot_to_png_bytes(title: str, x: list[float], y: list[float]) -> bytes:
    plt.figure(figsize=(5, 3), dpi=130)
    plt.plot(x, y, color="#003057")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf.read()


def build_demo_pdf_report(equipment_name: str, status: str) -> bytes:
    """Формирует простой PDF‑отчёт по демо‑данным с рекомендациями и графиками."""
    series = _demo_series(status)

    # Рекомендации по статусу
    if status == "crit":
        recs = [
            "Немедленно остановить оборудование и провести диагностику.",
            "Проверить узлы подшипников и балансировку ротора.",
            "Запланировать внеплановый ремонт и анализ вибраций." 
        ]
    elif status == "warn":
        recs = [
            "Провести внеочередной осмотр и смазку подшипников.",
            "Усилить мониторинг: увеличить частоту съёма данных.",
            "Проверить выравнивание и крепления узлов." 
        ]
    else:
        recs = [
            "Отклонений не обнаружено. Продолжать плановый мониторинг.",
            "Рекомендуется регулярная проверка креплений и виброуровня.",
        ]

    # Генерируем изображения графиков
    charts = []
    for name, (x, y) in series.items():
        charts.append((name, _plot_to_png_bytes(name, x, y)))

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="Отчёт по оборудованию", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, txt=f"Оборудование: {equipment_name}", ln=1)
    pdf.cell(0, 8, txt=f"Статус: {'Критично' if status=='crit' else ('Требует внимания' if status=='warn' else 'Все хорошо')}", ln=1)
    pdf.cell(0, 8, txt=f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, txt="Рекомендации:", ln=1)
    pdf.set_font("Arial", size=12)
    for r in recs:
        pdf.multi_cell(0, 6, txt=f"• {r}")

    # Вставляем графики (по два на страницу)
    for i, (name, img_bytes) in enumerate(charts):
        if i % 2 == 0:
            pdf.add_page()
            y_offset = 20
        else:
            y_offset = 150
        pdf.set_font("Arial", "B", 12)
        pdf.text(x=10, y=y_offset - 5, txt=name)
        pdf.image(io.BytesIO(img_bytes), x=10, y=y_offset, w=190)

    out = pdf.output(dest="S").encode("latin1", errors="ignore")
    return out
