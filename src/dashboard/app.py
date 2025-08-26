"""
Dashboard для диагностики асинхронных двигателей
Streamlit приложение с JWT авторизацией, визуализацией сигналов и отчетами
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64
import os
import logging

# Импорты локальных модулей
from .utils import (
    SessionManager, DataCache, DataProcessor,
    ValidationUtils, FormatUtils, SecurityUtils, ConfigManager
)
from .components import (
    UIComponents, ChartComponents, FilterComponents,
    ExportComponents, NotificationComponents, ConfigComponents
)
from .pages import AdminPage, render_trends

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация приложения
config = ConfigManager.load_dashboard_config()
API_BASE_URL = config.get('api_base_url', "http://api:8000")

st.set_page_config(
    page_title="DiagMod - Диагностика двигателей",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=20, show_spinner=False)
def api_ping(base_url: str) -> Dict:
    """Проверка доступности API: /health и / (root). Возвращает словарь статуса."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200 and isinstance(r.json(), dict):
            data = r.json()
            return {"ok": data.get("status") == "healthy", "endpoint": "/health", "details": data}
    except Exception as e:
        pass
    try:
        r = requests.get(f"{base_url}/", timeout=5)
        if r.status_code == 200 and isinstance(r.json(), dict):
            data = r.json()
            return {"ok": data.get("status") == "healthy", "endpoint": "/", "details": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False}

# Глобальный статус API и подсказки по Grafana в сайдбаре
try:
    st.sidebar.header("Состояние")
    ping = api_ping(API_BASE_URL)
    if ping.get("ok"):
        st.sidebar.success(f"API доступно ({ping.get('endpoint')})")
    else:
        st.sidebar.error("API недоступно")
    # Подсказки по Grafana
    st.sidebar.header("Grafana")
    g_base = os.getenv("GRAFANA_URL", "")
    g_path = os.getenv("GRAFANA_DASHBOARD_PATH", "/d/diag/overview")
    st.sidebar.caption("Переменные окружения:")
    st.sidebar.code(f"GRAFANA_URL={g_base or '<не задано>'}\nGRAFANA_DASHBOARD_PATH={g_path}")
    if not g_base:
        st.sidebar.info("Установите GRAFANA_URL для корректной встройки панели (например, http://localhost:3000)")
except Exception:
    pass

class AuthManager:
    """Менеджер авторизации через JWT"""
    
    @staticmethod
    def login(username: str, password: str) -> Optional[Dict]:
        """Авторизация пользователя"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/auth/login",
                json={"username": username, "password": password}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка авторизации: {e}")
            return None
    
    @staticmethod
    def verify_token(token: str) -> bool:
        """Проверка валидности токена"""
        try:
            # Проверяем токен через /auth/me (доступен и на /api/v1/me)
            response = requests.get(
                f"{API_BASE_URL}/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            return response.status_code == 200
        except Exception:
            return False
    
    @staticmethod
    def get_user_info(token: str) -> Optional[Dict]:
        """Получение информации о пользователе"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

class DataManager:
    """Менеджер данных для работы с API"""
    
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def get_equipment_list(self) -> List[Dict]:
        """Получение списка оборудования"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/equipment",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения списка оборудования: {e}")
            return []
    
    def get_equipment_files(self, equipment_id: int) -> List[Dict]:
        """Получение списка сигналов (файлов) для оборудования через /api/v1/signals"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/signals",
                headers=self.headers,
                params={"equipment_id": equipment_id, "page": 1, "page_size": 100}
            )
            if response.status_code == 200:
                data = response.json() or {}
                # Возвращаем список сигналов как «файлов» для совместимости с UI
                return data.get("signals", [])
            return []
        except Exception as e:
            logger.error(f"Ошибка получения сигналов: {e}")
            return []
    
    def get_signal_data(self, raw_id: int) -> Optional[Dict]:
        """Получение данных сигнала"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/signals/{raw_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения сигнала: {e}")
            return None
    
    def get_anomalies(self, equipment_id: int) -> List[Dict]:
        """Получение аномалий для оборудования"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/anomalies/{equipment_id}",
                headers=self.headers
            )
            if response.status_code == 200:
                payload = response.json() or {}
                # Приводим к старому формату списка аномалий
                anomalies = []
                for a in payload.get("anomalies", []) or []:
                    anomalies.append({
                        "is_anomaly": True,
                        "created_at": a.get("detected_at"),
                        "defect_type": a.get("anomaly_type"),
                        "probability": a.get("confidence"),
                        "predicted_severity": a.get("severity"),
                        "model_name": a.get("model_name"),
                    })
                return anomalies
            return []
        except Exception as e:
            logger.error(f"Ошибка получения аномалий: {e}")
            return []
    
    def get_features(self, raw_id: int) -> Optional[Dict]:
        """Получение признаков сигнала"""
        try:
            # Эндпоинта features для raw_id может не быть; оставляем как опционально
            response = requests.get(
                f"{API_BASE_URL}/api/v1/signals/{raw_id}/embeddings",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения признаков: {e}")
            return None

    def get_hourly_rms(self, equipment_id: int, limit: int = 168) -> List[Dict]:
        """Почасовые агрегаты RMS для трендов"""
        try:
            r = requests.get(
                f"{API_BASE_URL}/api/v1/equipment/{equipment_id}/rms/hourly",
                headers=self.headers,
                params={"limit": limit},
                timeout=30
            )
            if r.status_code == 200:
                return (r.json() or {}).get("points", [])
            return []
        except Exception as e:
            logger.error(f"Ошибка загрузки RMS агрегатов: {e}")
            return []

class Visualizer:
    """Класс для создания визуализаций"""
    
    @staticmethod
    def plot_time_series(signal_data: Dict, title: str = "Токовые сигналы"):
        """Построение временных рядов под схему /api/v1/signals/{id}"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Фаза R', 'Фаза S', 'Фаза T'),
            vertical_spacing=0.05
        )

        # Создание временной оси
        sample_rate = signal_data.get('sample_rate') or signal_data.get('sample_rate_hz') or 25600
        samples_count = signal_data.get('total_samples') or signal_data.get('samples_count') or 0
        time_axis = np.linspace(0, samples_count / sample_rate, samples_count) if samples_count and sample_rate else np.arange(0)

        # Данные фаз: либо поля phase_a/b/c, либо список phases
        phases_map = { 'A': [], 'B': [], 'C': [] }
        if 'phases' in signal_data and isinstance(signal_data['phases'], list):
            for p in signal_data['phases']:
                name = str(p.get('phase_name', '')).upper()
                vals = p.get('values') or []
                if name in phases_map:
                    phases_map[name] = np.array(vals)
        else:
            # fallback на старую схему
            for k, dest in [('phase_a','A'), ('phase_b','B'), ('phase_c','C')]:
                if k in signal_data:
                    phases_map[dest] = np.array(signal_data.get(k) or [])

        colors = ['red', 'green', 'blue']
        for i, (phase_key, color) in enumerate(zip(['A','B','C'], colors)):
            phase_data = phases_map.get(phase_key) or np.array([])
            if phase_data.size > 0:
                x = time_axis[:len(phase_data)] if time_axis.size else np.arange(len(phase_data)) / sample_rate
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=phase_data,
                        name=f'Фаза {phase_key}',
                        line=dict(color=color, width=1),
                        showlegend=True
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(title=title, height=800, showlegend=True)
        fig.update_xaxes(title_text="Время (с)", row=3, col=1)
        fig.update_yaxes(title_text="Ток (А)")
        return fig
    
    @staticmethod
    def plot_fft_spectrum(signal_data: Dict, features: Dict):
        """Построение FFT спектра"""
        if not features or 'fft_spectrum' not in features:
            return None
        
        fft_data = features['fft_spectrum']
        if not fft_data:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('FFT Фаза R', 'FFT Фаза S', 'FFT Фаза T'),
            vertical_spacing=0.05
        )
        
        phases = ['phase_a', 'phase_b', 'phase_c']
        colors = ['red', 'green', 'blue']
        
        for i, (phase, color) in enumerate(zip(phases, colors)):
            if phase in fft_data:
                spectrum = fft_data[phase]
                frequencies = spectrum.get('frequencies', [])
                magnitudes = spectrum.get('magnitudes', [])
                
                if frequencies and magnitudes:
                    fig.add_trace(
                        go.Scatter(
                            x=frequencies,
                            y=magnitudes,
                            name=f'FFT {phases[i][-1].upper()}',
                            line=dict(color=color, width=1)
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            title="Частотный спектр (FFT)",
            height=800
        )
        fig.update_xaxes(title_text="Частота (Гц)", row=3, col=1)
        fig.update_yaxes(title_text="Амплитуда")
        
        return fig
    
    @staticmethod
    def plot_rms_trend(anomalies: List[Dict]):
        """График тренда RMS и аномалий"""
        if not anomalies:
            return None
        
        # Извлечение данных RMS по времени
        rms_data = []
        for anomaly in anomalies:
            if 'rms_values' in anomaly:
                rms_data.extend(anomaly['rms_values'])
        
        if not rms_data:
            return None
        
        df = pd.DataFrame(rms_data)
        
        fig = go.Figure()
        
        # График RMS по фазам
        for phase in ['rms_a', 'rms_b', 'rms_c']:
            if phase in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'] if 'timestamp' in df.columns else range(len(df)),
                        y=df[phase],
                        name=f'RMS {phase[-1].upper()}',
                        mode='lines'
                    )
                )
        
        # Отметка аномалий
        anomaly_times = [a['detected_at'] for a in anomalies if a.get('is_anomaly')]
        if anomaly_times:
            for time in anomaly_times:
                fig.add_vline(
                    x=time,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Аномалия"
                )
        
        fig.update_layout(
            title="Тренд RMS и обнаруженные аномалии",
            xaxis_title="Время",
            yaxis_title="RMS (А)",
            height=500
        )
        
        return fig

class ReportGenerator:
    """Генератор PDF отчетов"""
    
    @staticmethod
    def generate_report(equipment_data: Dict, anomalies: List[Dict], 
                       signal_data: Dict, features: Dict) -> bytes:
        """Генерация PDF отчета"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Заголовок отчета
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # По центру
        )
        
        story.append(Paragraph("Отчет диагностики асинхронного двигателя", title_style))
        story.append(Spacer(1, 12))
        
        # Информация об оборудовании
        equipment_info = [
            ['Параметр', 'Значение'],
            ['ID оборудования', str(equipment_data.get('id', 'N/A'))],
            ['Название', equipment_data.get('name', 'N/A')],
            ['Модель', equipment_data.get('model', 'N/A')],
            ['Дата анализа', datetime.now().strftime('%d.%m.%Y %H:%M')]
        ]
        
        equipment_table = Table(equipment_info)
        equipment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(equipment_table)
        story.append(Spacer(1, 20))
        
        # Статистика признаков
        if features:
            story.append(Paragraph("Статистические характеристики", styles['Heading2']))
            
            features_info = []
            phases = ['a', 'b', 'c']
            
            for phase in phases:
                rms_val = features.get(f'rms_{phase}', 'N/A')
                crest_val = features.get(f'crest_{phase}', 'N/A')
                kurt_val = features.get(f'kurt_{phase}', 'N/A')
                skew_val = features.get(f'skew_{phase}', 'N/A')
                
                features_info.extend([
                    [f'Фаза {phase.upper()}', '', '', ''],
                    ['RMS', f'{rms_val:.3f}' if isinstance(rms_val, (int, float)) else str(rms_val), '', ''],
                    ['Crest Factor', f'{crest_val:.3f}' if isinstance(crest_val, (int, float)) else str(crest_val), '', ''],
                    ['Kurtosis', f'{kurt_val:.3f}' if isinstance(kurt_val, (int, float)) else str(kurt_val), '', ''],
                    ['Skewness', f'{skew_val:.3f}' if isinstance(skew_val, (int, float)) else str(skew_val), '', '']
                ])
            
            features_table = Table(features_info)
            story.append(features_table)
            story.append(Spacer(1, 20))
        
        # Результаты диагностики
        story.append(Paragraph("Результаты диагностики", styles['Heading2']))
        
        if anomalies:
            anomaly_count = len([a for a in anomalies if a.get('is_anomaly')])
            story.append(Paragraph(f"Обнаружено аномалий: {anomaly_count}", styles['Normal']))
            
            if anomaly_count > 0:
                anomaly_data = [['Время обнаружения', 'Тип', 'Вероятность', 'Серьезность']]
                for anomaly in anomalies:
                    if anomaly.get('is_anomaly'):
                        anomaly_data.append([
                            anomaly.get('detected_at', 'N/A'),
                            anomaly.get('defect_type', 'Неизвестно'),
                            f"{anomaly.get('probability', 0):.3f}",
                            anomaly.get('predicted_severity', 'N/A')
                        ])
                
                anomaly_table = Table(anomaly_data)
                anomaly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(anomaly_table)
        else:
            story.append(Paragraph("Аномалий не обнаружено", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Рекомендации
        story.append(Paragraph("Рекомендации", styles['Heading2']))
        if anomalies and any(a.get('is_anomaly') for a in anomalies):
            recommendations = [
                "• Провести детальную диагностику выявленных аномалий",
                "• Увеличить частоту мониторинга состояния двигателя",
                "• Рассмотреть возможность планового технического обслуживания",
                "• Проанализировать условия эксплуатации оборудования"
            ]
        else:
            recommendations = [
                "• Двигатель работает в нормальном режиме",
                "• Рекомендуется продолжить регулярный мониторинг",
                "• Следующая диагностика через установленный интервал"
            ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

def main():
    """Основная функция приложения"""
    
    # Заголовок приложения
    st.title("⚡ DiagMod - Система диагностики асинхронных двигателей")
    
    # Проверка авторизации
    if 'token' not in st.session_state or not st.session_state.get('token'):
        show_login()
        return
    
    # Проверка валидности токена
    if not AuthManager.verify_token(st.session_state.token):
        st.error("Сессия истекла. Пожалуйста, авторизуйтесь заново.")
        st.session_state.clear()
        st.rerun()
        return
    
    # Получение информации о пользователе
    user_info = AuthManager.get_user_info(st.session_state.token)
    if not user_info:
        st.error("Ошибка получения информации о пользователе")
        return

    # Устанавливаем режим просмотра по умолчанию
    if 'view' not in st.session_state:
        st.session_state.view = 'home'
    
    # Боковая панель с информацией о пользователе
    with st.sidebar:
        st.write(f"👤 Пользователь: {user_info.get('username', 'N/A')}")
        st.write(f"🔒 Роль: {user_info.get('role', 'N/A')}")
        
        # Кнопка возврата на главную
        if st.button("🏠 Вернуться на главную"):
            st.session_state.view = 'home'
            st.session_state.pop('selected_equipment_id', None)
            st.rerun()

        if st.button("Выйти"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()

        # Небольшое окно настроек (popover если доступен, иначе expander)
        popover = getattr(st, "popover", None)
        if callable(popover):
            with st.popover("⚙️ Настройки"):
                st.write("Быстрые действия")
                if st.button("Настроить уведомления", key="configure_notifications"):
                    st.success("Окно настроек уведомлений: функция будет доработана.")
        else:
            with st.expander("⚙️ Настройки"):
                st.write("Быстрые действия")
                if st.button("Настроить уведомления", key="configure_notifications_exp"):
                    st.success("Окно настроек уведомлений: функция будет доработана.")

        # Кнопка генерации отчета доступна в режиме детали двигателя
        if st.session_state.view != 'home' and st.session_state.get('selected_equipment_id'):
            eq_id = st.session_state.get('selected_equipment_id')
            st.markdown("---")
            st.subheader("📄 Отчет")
            if st.button("Сгенерировать отчет", type="primary", key="generate_report_sidebar"):
                try:
                    dm = DataManager(st.session_state.token)
                    anomalies = dm.get_anomalies(eq_id)
                    files = dm.get_equipment_files(eq_id)
                    signal_data, features = {}, {}
                    if files:
                        # Берем последний по времени файл
                        try:
                            latest = max(files, key=lambda x: x.get('recorded_at', '') or '')
                        except Exception:
                            latest = files[0]
                        raw_id = latest.get('raw_signal_id') or latest.get('id')
                        if raw_id is not None:
                            signal_data = dm.get_signal_data(raw_id) or {}
                            features = dm.get_features(raw_id) or {}
                    # Для заголовка отчета нужна базовая инфа об оборудовании
                    # Получим список и найдем запись по id
                    equipment_list = dm.get_equipment_list()
                    eq_data = next((e for e in equipment_list if e.get('id') == eq_id), {"id": eq_id, "name": f"Двигатель {eq_id}"})

                    pdf_bytes = ReportGenerator.generate_report(eq_data, anomalies, signal_data, features)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"diagmod_report_{eq_id}_{ts}.pdf"
                    # Сохраним в состоянии для отображения кнопки скачивания
                    st.session_state['last_report_pdf'] = pdf_bytes
                    st.session_state['last_report_name'] = file_name
                    st.success("Отчет сформирован")
                except Exception as e:
                    st.error(f"Не удалось сформировать отчет: {e}")

            # Если есть последний отчет в состоянии — показать кнопку скачивания
            if st.session_state.get('last_report_pdf'):
                st.download_button(
                    label="📥 Скачать отчет",
                    data=st.session_state['last_report_pdf'],
                    file_name=st.session_state.get('last_report_name', 'report.pdf'),
                    mime='application/pdf',
                    key="download_report_sidebar"
                )
    
    # Маршрутизация между главной страницей и деталями двигателя
    if st.session_state.view == 'home':
        show_home()
    else:
        show_dashboard(preselect_equipment_id=st.session_state.get('selected_equipment_id'))

def show_login():
    """Форма авторизации"""
    st.subheader("🔐 Авторизация")
    
    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")
        
        if submitted:
            if username and password:
                auth_result = AuthManager.login(username, password)
                if auth_result and 'access_token' in auth_result:
                    st.session_state.token = auth_result['access_token']
                    st.session_state.user_info = auth_result.get('user_info', {})
                    st.success("Авторизация успешна!")
                    st.rerun()
                else:
                    st.error("Неверные учетные данные")
            else:
                st.error("Заполните все поля")

def show_dashboard(preselect_equipment_id: int | None = None):
    """Детальная страница двигателя (основной дашборд)"""
    data_manager = DataManager(st.session_state.token)
    
    # Получение списка оборудования
    equipment_list = data_manager.get_equipment_list()
    
    if not equipment_list:
        st.warning("Нет доступного оборудования или ошибка загрузки данных")
        return
    
    # Выбор оборудования
    equipment_options = {eq['name']: eq for eq in equipment_list}
    names = list(equipment_options.keys())
    # Определяем индекс по умолчанию, если задан preselect_equipment_id
    default_index = 0
    if preselect_equipment_id is not None:
        for i, name in enumerate(names):
            if equipment_options[name].get('id') == preselect_equipment_id:
                default_index = i
                break
    selected_equipment_name = st.selectbox(
        "🔧 Выберите оборудование:",
        options=names,
        index=default_index if names else 0
    )
    
    if not selected_equipment_name:
        return
    
    selected_equipment = equipment_options[selected_equipment_name]
    equipment_id = selected_equipment['id']
    # Сохраняем выбранный двигатель в состоянии (для возврата из страниц)
    st.session_state.selected_equipment_id = equipment_id
    
    # Информация об оборудовании
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ID оборудования", equipment_id)
    with col2:
        st.metric("Модель", selected_equipment.get('model', 'N/A'))
    with col3:
        st.metric("Статус", selected_equipment.get('status', 'N/A'))
    
    # Вкладки функциональности
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Обзор файлов", "📈 Анализ сигналов", "⚠️ Аномалии", "📄 Отчеты", "📈 Тренды"])
    
    with tab1:
        show_files_overview(data_manager, equipment_id)
    
    with tab2:
        show_signal_analysis(data_manager, equipment_id)
    
    with tab3:
        show_anomalies(data_manager, equipment_id)
    
    with tab4:
        show_reports(data_manager, equipment_id, selected_equipment)
    with tab5:
        render_trends(API_BASE_URL, st.session_state.token, str(equipment_id))

def _compute_status_label(anomalies_recent: int, severity_max: float | None, has_signals: bool) -> tuple[str, str]:
    """Возвращает (эмодзи, короткий статус) по данным.
    Правила:
    - ⚪ Нет сигналов → "Двигатель выключен / нет сигналов"
    - 🔴 Есть критические аномалии (severity >= 0.8) → "Критическая ошибка"
    - 🟠 Есть аномалии за 7д → "Требует внимания"
    - 🟢 Иначе → "Все ок"
    """
    if not has_signals:
        return "⚪", "Двигатель выключен / нет сигналов"
    if anomalies_recent > 0 and (severity_max is not None and severity_max >= 0.8):
        return "🔴", "Критическая ошибка"
    if anomalies_recent > 0:
        return "🟠", "Требует внимания"
    return "🟢", "Все ок"

def show_home():
    """Главная страница: список двигателей со светофором статуса и переходом по клику."""
    st.subheader("Главная: доступные двигатели")
    data_manager = DataManager(st.session_state.token)

    equipment_list = data_manager.get_equipment_list()
    if not equipment_list:
        st.warning("Нет доступного оборудования или ошибка загрузки данных")
        return

    # Подготовим агрегацию по статусам
    rows = []
    now = pd.Timestamp.utcnow()
    horizon = now - pd.Timedelta(days=7)

    for eq in equipment_list:
        eq_id = eq.get('id')
        eq_name = eq.get('name', f"Двигатель {eq_id}")
        eq_model = eq.get('model', '—')

        # Сигналы (для последней даты)
        files = data_manager.get_equipment_files(eq_id) or []
        last_ts = None
        if files:
            try:
                df_files = pd.DataFrame(files)
                if 'recorded_at' in df_files.columns:
                    ts = pd.to_datetime(df_files['recorded_at'], errors='coerce')
                    last_ts = ts.max()
            except Exception:
                last_ts = None

        has_signals = bool(files)

        # Аномалии за 7 дней
        anomalies = data_manager.get_anomalies(eq_id) or []
        anomalies_recent = 0
        severity_max = None
        for a in anomalies:
            try:
                created = a.get('created_at') or a.get('detected_at')
                created_ts = pd.to_datetime(created, errors='coerce') if created else None
                if a.get('is_anomaly') and (created_ts is None or created_ts >= horizon):
                    anomalies_recent += 1
                    sev = a.get('predicted_severity')
                    if isinstance(sev, (int, float)):
                        severity_max = max(severity_max or 0.0, float(sev))
            except Exception:
                pass

        emoji, short = _compute_status_label(anomalies_recent, severity_max, has_signals)
        rows.append({
            'ID': eq_id,
            'Двигатель': eq_name,
            'Модель': eq_model,
            'Последний сигнал': last_ts.tz_localize(None).strftime('%Y-%m-%d %H:%M') if isinstance(last_ts, pd.Timestamp) and pd.notnull(last_ts) else '—',
            'Аномалий (7д)': anomalies_recent,
            'Статус': f"{emoji} {short}",
        })

    # Отображаем таблицу
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Переход к двигателю")
    for row in rows:
        c1, c2, c3, c4, c5, c6 = st.columns([0.12, 0.45, 0.18, 0.15, 0.15, 0.15])
        with c1:
            st.write(row['Статус'])
        with c2:
            st.write(f"{row['Двигатель']} (ID: {row['ID']})")
        with c3:
            st.write(row['Модель'])
        with c4:
            st.write(row['Последний сигнал'])
        with c5:
            st.write(f"Аномалий: {row['Аномалий (7д)']}")
        with c6:
            if st.button("Открыть", key=f"open_{row['ID']}"):
                st.session_state.selected_equipment_id = row['ID']
                st.session_state.view = 'detail'
                st.rerun()

def show_files_overview(data_manager: DataManager, equipment_id: int):
    """Обзор файлов оборудования"""
    st.subheader("📊 Обзор файлов данных")
    
    files = data_manager.get_equipment_files(equipment_id)
    
    if not files:
        st.info("Файлы данных не найдены")
        return
    
    # Статистика файлов
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего файлов", len(files))
    with col2:
        total_samples = sum(f.get('samples_count', 0) for f in files)
        st.metric("Всего образцов", f"{total_samples:,}")
    with col3:
        # Длительность недоступна из API списка сигналов — скрываем метрику
        st.metric("Средняя длительность", "—")
    with col4:
        latest_file = max(files, key=lambda x: x.get('recorded_at', ''), default={})
        latest_date = latest_file.get('recorded_at', 'N/A')
        st.metric("Последний файл", latest_date[:10] if latest_date != 'N/A' else 'N/A')
    
    # Таблица файлов
    if files:
        # Адаптируем под фактические поля API списка сигналов
        df_files = pd.DataFrame(files)
        cols = [c for c in ['raw_signal_id', 'recorded_at', 'samples_count', 'processing_status'] if c in df_files.columns]
        st.dataframe(df_files[cols], use_container_width=True)

def show_signal_analysis(data_manager: DataManager, equipment_id: int):
    """Анализ сигналов"""
    st.subheader("📈 Анализ токовых сигналов")
    
    files = data_manager.get_equipment_files(equipment_id)
    
    if not files:
        st.info("Файлы для анализа не найдены")
        return
    
    # Выбор файла для анализа
    file_options = {f"Файл {f.get('raw_signal_id', f.get('id'))} ({str(f.get('recorded_at', ''))[:19]})": (f.get('raw_signal_id') or f.get('id')) for f in files}
    selected_file = st.selectbox("Выберите файл для анализа:", options=list(file_options.keys()))
    
    if not selected_file:
        return
    
    raw_id = file_options[selected_file]
    
    # Загрузка данных сигнала
    with st.spinner("Загрузка данных сигнала..."):
        signal_data = data_manager.get_signal_data(raw_id)
        features = data_manager.get_features(raw_id)
    
    if not signal_data:
        st.error("Ошибка загрузки данных сигнала")
        return
    
    # Визуализация временных рядов
    st.subheader("Временные ряды токов")
    fig_time = Visualizer.plot_time_series(signal_data)
    if fig_time:
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Визуализация FFT спектра
    if features and isinstance(features, dict):
        st.subheader("Частотный анализ (FFT)")
        fig_fft = Visualizer.plot_fft_spectrum(signal_data, features)
        if fig_fft:
            st.plotly_chart(fig_fft, use_container_width=True)
        
        # Статистические характеристики
        st.subheader("Статистические характеристики")
        col1, col2, col3 = st.columns(3)
        
        phases = ['a', 'b', 'c']
        phase_names = ['R', 'S', 'T']
        
        for i, (phase, name) in enumerate(zip(phases, phase_names)):
            with [col1, col2, col3][i]:
                st.write(f"**Фаза {name}:**")
                rms_val = features.get(f'rms_{phase}')
                crest_val = features.get(f'crest_{phase}')
                kurt_val = features.get(f'kurt_{phase}')
                skew_val = features.get(f'skew_{phase}')
                
                if rms_val is not None:
                    st.metric("RMS", f"{rms_val:.3f}")
                if crest_val is not None:
                    st.metric("Crest Factor", f"{crest_val:.3f}")
                if kurt_val is not None:
                    st.metric("Kurtosis", f"{kurt_val:.3f}")
                if skew_val is not None:
                    st.metric("Skewness", f"{skew_val:.3f}")

def show_anomalies(data_manager: DataManager, equipment_id: int):
    """Отображение аномалий и прогнозов"""
    st.subheader("⚠️ Диагностика и прогнозирование")

    # Загрузка данных аномалий
    with st.spinner("Загрузка данных диагностики..."):
        anomalies = data_manager.get_anomalies(equipment_id)
    
    if not anomalies:
        st.info("Данные диагностики не найдены")
        return
    
    # Общая статистика
    anomaly_count = len([a for a in anomalies if a.get('is_anomaly')])
    total_predictions = len(anomalies)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего прогнозов", total_predictions)
    with col2:
        st.metric("Обнаружено аномалий", anomaly_count, delta=f"{anomaly_count/total_predictions*100:.1f}%" if total_predictions > 0 else "0%")
    with col3:
        last_check = max([a.get('created_at', '') for a in anomalies], default='N/A')
        st.metric("Последняя проверка", last_check[:19] if last_check != 'N/A' else 'N/A')
    
    # График тренда RMS (почасовые агрегаты) с отметками аномалий
    rms_points = data_manager.get_hourly_rms(equipment_id, limit=24*7)
    if rms_points:
        try:
            df = pd.DataFrame(rms_points)
            ts_col = 'timestamp' if 'timestamp' in df.columns else ('ts' if 'ts' in df.columns else None)
            if ts_col:
                df['ts'] = pd.to_datetime(df[ts_col])
                y_col = 'rms_mean' if 'rms_mean' in df.columns else None
                if y_col:
                    fig = px.line(df.sort_values('ts'), x='ts', y=y_col, title='RMS (среднее) по часам')
                    # Отметим времена аномалий
                    anomaly_times = [a.get('created_at') for a in anomalies if a.get('is_anomaly') and a.get('created_at')]
                    import plotly.graph_objects as go
                    for t in anomaly_times:
                        try:
                            fig.add_vline(x=pd.to_datetime(t), line_dash='dash', line_color='red')
                        except Exception:
                            pass
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось построить тренд RMS: {e}")
    
    # Таблица аномалий
    if anomaly_count > 0:
        st.subheader("Обнаруженные аномалии")
        anomaly_data = []
        for anomaly in anomalies:
            if anomaly.get('is_anomaly'):
                anomaly_data.append({
                    'Время': anomaly.get('created_at', 'N/A')[:19],
                    'Тип дефекта': anomaly.get('defect_type', 'Неизвестно'),
                    'Вероятность': f"{anomaly.get('probability', 0):.3f}",
                    'Серьезность': anomaly.get('predicted_severity', 'N/A'),
                    'Модель': anomaly.get('model_name', 'N/A')
                })
        
        if anomaly_data:
            df_anomalies = pd.DataFrame(anomaly_data)
            st.dataframe(df_anomalies, use_container_width=True)
    else:
        st.success("🟢 Аномалий не обнаружено. Оборудование работает в нормальном режиме.")

def show_reports(data_manager: DataManager, equipment_id: int, equipment_data: Dict):
    """Генерация отчетов"""
    st.subheader("📄 Генерация отчетов")
    
    # Выбор параметров отчета
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Тип отчета:",
            ["Полный диагностический отчет", "Отчет по аномалиям", "Сводка по оборудованию"]
        )
    
    with col2:
        include_charts = st.checkbox("Включить графики", value=True)
    
    # Получение данных для отчета
    if st.button("Сгенерировать отчет", type="primary"):
        with st.spinner("Генерация отчета..."):
            try:
                # Загрузка всех необходимых данных
                anomalies = data_manager.get_anomalies(equipment_id)
                files = data_manager.get_equipment_files(equipment_id)
                
                # Получение последнего файла для анализа
                signal_data = {}
                features = {}
                if files:
                    latest_file = max(files, key=lambda x: x.get('recorded_at', ''), default={})
                    if latest_file:
                        signal_data = data_manager.get_signal_data(latest_file['id']) or {}
                        features = data_manager.get_features(latest_file['id']) or {}
                
                # Генерация PDF отчета
                pdf_bytes = ReportGenerator.generate_report(
                    equipment_data, anomalies, signal_data, features
                )
                
                # Создание ссылки для скачивания
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"diagmod_report_{equipment_id}_{timestamp}.pdf"
                
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">📥 Скачать отчет</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Отчет успешно сгенерирован!")
                
            except Exception as e:
                st.error(f"Ошибка генерации отчета: {e}")
                logger.error(f"Ошибка генерации отчета: {e}")

if __name__ == "__main__":
    main()
