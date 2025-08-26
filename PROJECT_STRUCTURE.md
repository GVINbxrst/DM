# Структура проекта токовой диагностики асинхронных двигателей

## 1. Полная иерархия (актуально для репозитория)

```
DiagnosticModel/
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/            # endpoints (auth, signals, anomalies, monitoring, ...)
│   │   └── middleware/
│   ├── config/
│   │   ├── settings.py
│   │   └── logging.py
│   ├── database/
│   │   ├── connection.py
│   │   └── models.py
│   ├── data_processing/
│   │   ├── csv_loader.py
│   │   ├── data_validator.py
│   │   └── feature_extraction.py
│   ├── ml/
│   │   ├── forecasting.py
│   │   ├── tcn_forecasting.py
│   │   ├── clustering.py
│   │   ├── data_readiness.py
│   │   └── utils.py
│   ├── worker/
│   │   ├── tasks.py
│   │   ├── tasks_logic.py
│   │   └── processing_core.py
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       ├── feature_store.py
│       └── prediction_cache.py
│
├── alembic/
│   ├── env.py
│   └── versions/
│
├── sql/
│   ├── schema/
│   ├── procedures/
│   └── views/
│
├── docker/
│   ├── api/Dockerfile
│   ├── worker/Dockerfile
│   ├── dashboard/Dockerfile
│   └── jupyter/Dockerfile
│
├── configs/
│   ├── nginx/
│   ├── grafana/
│   ├── prometheus/
│   ├── postgres/
│   └── redis/
│
├── data/
├── models/
├── scripts/
└── tests/
```

## 2. Назначение папок и файлов

### src/
- api: приложение FastAPI (`main.py`, `routes/*`, middleware, schemas)
- worker: Celery и задачи обработки/ML
- dashboard: Streamlit UI
- ml: прогнозы (Prophet/sequence/TCN), кластеризация, утилиты
- data_processing: `csv_loader.py`, валидация и извлечение признаков
- database: подключение и модели SQLAlchemy
- utils: логирование, метрики, кеширование, feature store

### data/ - Данные
- **raw/**: Исходные CSV файлы по двигателям
- **processed/**: Очищенные и подготовленные данные
- **features/**: Извлеченные признаки для ML
- **exports/**: Экспорты для анализа

### models/
- Хранилище моделей: `prophet_rms/`, `tcn/`, `anomaly_detection/` и манифесты (см. README)

### sql/ - База данных
- **schema/**: DDL скрипты создания таблиц
- **procedures/**: Хранимые процедуры для сложной логики
- **views/**: Представления для отчетности
- **seed/**: Начальные данные

### scripts/ - Автоматизация
- Скрипты для развертывания, загрузки данных, обучения
- Утилиты администрирования и мониторинга

## 3. Расположение компонентов

### SQL схемы
- `sql/schema/` - DDL скрипты
- `src/database/models.py` - SQLAlchemy модели
- `src/database/migrations/` - Alembic миграции

### Загрузка CSV
- `src/data_processing/csv_loader.py` — основной модуль
- Docker сервис `loader` (см. `docker-compose.dev.yml`) — разовые загрузки локальных CSV

### Извлечение признаков
- `src/data_processing/feature_extractor.py` - FFT, RMS, kurtosis, skewness
- `src/data_processing/signal_processing.py` - обработка сигналов
- `src/ml/feature_engineering.py` - продвинутая инженерия признаков

### ML-модели и прогнозирование
- `src/ml/anomaly_detection.py` - модели аномалий
- `src/ml/prediction.py` - прогнозные модели
- `src/ml/model_training.py` - обучение
- `models/` - хранение обученных моделей

### API и роуты
- `src/api/main.py` — FastAPI приложение, `/health`, регистрация роутов
- `src/api/routes/` — `signals.py`, `upload.py`, `anomalies.py`, `auth.py`, `monitoring.py`, `admin_*`

### Фоновые задачи Celery
- Redis — брокер, результаты в Redis (см. compose)
- Worker запускается из Docker (`docker/worker/Dockerfile`) или локально (Makefile)

### Дашборд Streamlit
- `src/dashboard/main.py`, `pages/`, компоненты; использует API

### Конфиги
- Python-конфиги: `src/config/*`
- Файлы: `configs/*` (nginx, grafana, prometheus, postgres, redis)

## 4. Взаимодействие модулей

```
CSV Files → Data Processing → Database → Feature Extraction → ML Models
                ↓                ↓              ↓              ↓
            Worker Tasks    → API Routes → Dashboard → Monitoring
                ↓                ↓              ↓              ↓
            Celery Queue    → Redis Cache → Streamlit → Prometheus/Grafana
```

**Поток данных:**
1. CSV файлы загружаются через `csv_loader.py`
2. Данные валидируются и сохраняются в PostgreSQL
3. Worker задачи извлекают признаки асинхронно
4. ML модели обучаются на признаках
5. API предоставляет доступ к данным и прогнозам
6. Dashboard отображает результаты
7. Prometheus собирает метрики, Grafana визуализирует

**Связи между компонентами:**
- Database модели используются во всех модулях
- Config модули импортируются везде
- Utils предоставляют общую функциональность
- API вызывает ML модели и Worker задачи
- Dashboard обращается к API
- Все компоненты логируют в единую систему

## 5. Запуск через Docker

```yaml
# docker-compose.dev.yml (основное для разработки)
services:
  postgres:
  redis:
  api:
  worker:
  dashboard:
  prometheus:
  grafana:
  jupyter:
  loader:   # для разовых загрузок CSV
```

**Быстрый запуск (dev):**
1. `docker compose -f docker-compose.dev.yml up -d postgres redis`
2. `docker compose -f docker-compose.dev.yml up -d api worker dashboard`
3. (опц.) `docker compose -f docker-compose.dev.yml up -d prometheus grafana`

Полезные цели Makefile:
```bash
make docker-build   # сборка образов
make docker-up      # запуск по docker-compose.yml
make db-upgrade     # alembic upgrade head
make init-db        # скриптовая инициализация
make dev-api        # uvicorn --reload
make dev-worker     # celery worker
make dev-dashboard  # streamlit
make test           # pytest с покрытием
```

## 6. Хранение и именование моделей

**Структура версионирования:**
```
models/
├── anomaly_detection/
│   ├── v1.0.0/
│   │   ├── model.pkl          # Основная модель
│   │   ├── scaler.pkl         # Препроцессор
│   │   ├── metadata.json      # Метаданные (метрики, дата)
│   │   └── config.yaml        # Гиперпараметры
│   ├── v1.1.0/               # Следующая версия
│   └── latest -> v1.1.0/     # Симлинк на актуальную
├── prediction/
└── registry.json             # Глобальный реестр моделей
```

**Именование:**
- Семантическое версионирование (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes в API модели
- MINOR: Новая функциональность, обратно совместимо
- PATCH: Исправления багов

**Метаданные модели:**
```json
{
  "version": "1.0.0",
  "created_at": "2025-01-10T10:00:00Z",
  "model_type": "IsolationForest",
  "metrics": {
    "precision": 0.95,
    "recall": 0.87,
    "f1_score": 0.91
  },
  "features": ["rms_R", "fft_peak_R", "kurtosis_R"],
  "training_data_hash": "abc123...",
  "git_commit": "a1b2c3d4"
}
```

## 7. Зависимости

### Зависимости

Основные пакеты смотрите в `pyproject.toml` и `requirements*.txt`. Ключевые группы:
- Web/API: fastapi, uvicorn, streamlit, pydantic, pydantic-settings
- БД: sqlalchemy[asyncio], asyncpg, alembic, psycopg2-binary, aiosqlite
- Очереди/кеш: celery[redis], redis
- ML/DS: numpy, scipy, pandas, scikit-learn, statsmodels (+ extras: prophet, xgboost, torch и др.)
- Сигналы: librosa, pywavelets
- Мониторинг: prometheus-client, structlog
- Security: python-jose[cryptography], passlib[bcrypt]

### pyproject.toml (современный подход)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "diagmod"
version = "1.0.0"
description = "Токовая диагностика асинхронных двигателей"
authors = [{name = "DiagMod Team"}]
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    # ... остальные зависимости
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.7.0",
]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88
```

## 8. Диаграмма взаимодействия компонентов

```
                           ┌─────────────────┐
                           │   CSV Files     │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐
                           │  Data Loader    │
                           │  (csv_loader)   │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐      ┌─────────────────┐
                           │   PostgreSQL    │◄────►│   Redis Cache   │
                           │   (raw_data)    │      │   (sessions)    │
                           └─────────┬───────┘      └─────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │ Feature Extract │
                           │ (Celery Worker) │
                           └─────────┬───────┘
                                     │
                ┌────────────────────▼────────────────────┐
                │                ML Models                │
                ├─────────────┬───────────────────────────┤
                │ Anomaly     │ Prediction  │ Forecasting │
                │ Detection   │ Models      │ (Prophet)   │
                └─────────────┴─────────────┬─────────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │                FastAPI Server                     │
                ├─────────┬─────────┬─────────┬──────────┬─────────┤
                │ Auth    │ Data    │ Models  │ Diagnostics│ Monitor │
                │ Routes  │ Routes  │ Routes  │ Routes   │ Routes  │
                └─────────┴─────────┴─────────┴──────────┴─────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │              Streamlit Dashboard                  │
                ├─────────┬─────────┬─────────┬──────────┬─────────┤
                │ Data    │ Anomaly │ Trends  │ Motor    │ Admin   │
                │ Overview│ Detection│ Analysis│ Status   │ Panel   │
                └─────────┴─────────┴─────────┴──────────┴─────────┘
                                           │
                ┌─────────────────────────▼─────────────────────────┐
                │                 Monitoring                        │
                ├─────────────────┬───────────────────────────────────┤
                │   Prometheus    │           Grafana               │
                │   (Metrics)     │         (Dashboards)            │
                └─────────────────┴───────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            Security Layer                           │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┤
│ JWT Auth        │ Role-based      │ API Rate        │ Data            │
│ (API Access)    │ Access Control  │ Limiting        │ Encryption      │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Легенда:**
- `│` - прямая связь
- `◄────►` - двусторонняя связь
- `▼` - поток данных вниз
- Прямоугольники - компоненты системы
- Горизонтальные разделители - логические группы

**Описание потоков:**
1. **Данные**: CSV → Loader → PostgreSQL → Features → ML
2. **API**: ML Models → FastAPI → Dashboard → Users  
3. **Фоновые задачи**: PostgreSQL → Celery → ML Training
4. **Мониторинг**: All Components → Prometheus → Grafana
5. **Безопасность**: JWT → RBAC → Rate Limiting → Encryption

Эта структура обеспечивает:
- **Масштабируемость**: Микросервисная архитектура
- **Модульность**: Четкое разделение ответственности
- **Безопасность**: Многоуровневая защита
- **Мониторинг**: Полная наблюдаемость системы
- **Развертывание**: Docker-compose для легкого запуска
