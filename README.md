# DiagMod — система токовой диагностики асинхронных двигателей

## Описание

DiagMod — комплекс для диагностики асинхронных двигателей на основе анализа токовых сигналов и ML. Поддерживает загрузку CSV, извлечение признаков, детекцию аномалий, краткосрочный прогноз риска, REST API и Streamlit‑дашборд, мониторинг Prometheus/Grafana, асинхронную обработку в Celery.

## Основные возможности

- Анализ токовых сигналов: CSV с фазами R,S,T (строгий заголовок current_R,current_S,current_T)
- Извлечение признаков: RMS, FFT, статистики; агрегаты по часам для трендов
- ML: аномалии, риск по последовательностям (LSTM/TCN), прогноз RMS (Prophet)
- API на FastAPI, дашборд на Streamlit, Celery worker, Redis, PostgreSQL
- Мониторинг и алертинг: Prometheus + Grafana

## Архитектура (схематично)

CSV → data_processing → PostgreSQL → feature extraction → ML models
                               ↓              ↓             ↓                 ↓
                     Celery tasks → Redis cache → FastAPI → Streamlit → Prometheus/Grafana

## Быстрый старт

Полный пошаговый гайд по запуску на Ubuntu VM и локально (Windows/Linux): см. файл `RUNBOOK.md`.

### 1) Клонирование и настройка

```bash
git clone <repository-url>
cd DiagnosticModel
make setup    # создаст venv, .env и рабочие каталоги (Windows/*nix)
```

### 2) Конфигурация

`.env` генерируется `scripts/setup.ps1`. При необходимости поправьте DATABASE_URL, REDIS_URL, CELERY_* и пр.

### 3) Запуск через Docker

Вариант для разработки (hot‑reload):

```bash
docker compose -f docker-compose.dev.yml up -d postgres redis
docker compose -f docker-compose.dev.yml up -d api worker dashboard prometheus grafana
```

Опция Jupyter (не обязательно):

```bash
docker compose -f docker-compose.dev.yml up -d jupyter
```

Прод-ориентированный стек:

```bash
make docker-build
make docker-up
```

Миграции/инициализация БД:

```bash
make db-upgrade   # alembic upgrade head
make init-db      # скриптовая инициализация (scripts/init_db.py)
```

### 4) Доступ к сервисам

- API: http://localhost:8000 (docs: /docs)
- Dashboard: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

### 5) Локальный E2E (без Docker)

```bash
make install-dev
set CELERY_TASK_ALWAYS_EAGER=1   # PowerShell, для bash: export CELERY_TASK_ALWAYS_EAGER=1
make process-raw
make anomalies
make forecast
make report
make pipeline   # полный конвейер
```

### 6) Загрузка локальных CSV через Docker (loader)

В `docker-compose.dev.yml` есть сервис `loader` с примонтированной папкой `./data:/data:ro`.

```bash
docker compose -f docker-compose.dev.yml up -d postgres redis
docker compose -f docker-compose.dev.yml run --rm loader `
     python src/data_processing/csv_loader.py /data/sample.csv --sample-rate 25600

# Загрузка всех файлов из каталога data/
docker compose -f docker-compose.dev.yml run --rm loader `
     python src/data_processing/csv_loader.py /data --sample-rate 25600 --batch-size 10000
```

Примечание: формат CSV строгий — первая строка заголовка: `current_R,current_S,current_T`. Пустые значения допустимы для S/T.

## API (фрагменты)

- GET /health — статус сервиса и БД
- GET /api/v1/signals — список сырьевых сигналов (авторизация)
- GET /api/v1/signals/{raw_id} — данные по сигналу, опции превью/прореживание
- GET /api/v1/equipment/{id}/rms/hourly?limit=168 — почасовой тренд RMS
- GET /api/v1/signals/sequence_risk/{equipment_id} — краткосрочный риск по последовательности
- /auth/* — аутентификация, обновление/отзыв сессий
- /monitoring/* — метрики и служебные проверки

## Структура проекта (сокращённо)

```
DiagnosticModel/
├─ src/
│  ├─ api/ (main.py, routes/*, middleware/*, schemas/*)
│  ├─ worker/ (celery, tasks, обработка пайплайна)
│  ├─ dashboard/ (Streamlit)
│  ├─ ml/ (forecasting, tcn_forecasting, clustering, utils)
│  ├─ data_processing/ (csv_loader.py, feature_extraction.py, data_validator.py)
│  ├─ database/ (connection.py, models.py)
│  └─ utils/ (logger.py, metrics.py, feature_store.py, ...)
├─ sql/ (schema/, views/, procedures/, seed/)
├─ alembic/ (env.py, versions/*)
├─ docker/ (api/, worker/, dashboard/, jupyter/)
├─ configs/ (nginx/, grafana/, prometheus/, postgres/, redis/)
├─ data/  (локальные CSV/артефакты)
├─ models/ (prophet_rms/, tcn/, ...)
├─ scripts/ (ingest/e2e/analytics/report/health и пр.)
├─ tests/ (unit, integration, fixtures)
├─ docker-compose.dev.yml
├─ docker-compose.yml
└─ Makefile
```

## Формат данных (CSV)

```
current_R,current_S,current_T
1.23,1.45,1.67
1.24,,1.68
1.25,1.47,
...
```

## Разработка

Установка зависимостей:

```bash
make install-dev
```

Локальный Postgres (без Docker):

```
postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod

# PowerShell
$env:DATABASE_URL = "postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod"
$env:APP_ENVIRONMENT = "development"
```

Локальный запуск без Docker: см. `LOCAL_RUN.md`.

Тесты и качество кода:

```bash
make test
make format
make lint
```

Режим разработки (hot‑reload):

```bash
make dev-api       # Uvicorn --reload
make dev-worker    # Celery worker
make dev-dashboard # Streamlit
```

## Мониторинг и производительность

- Prometheus метрики (API: порт 8001), дашборды Grafana
- Партиционирование таблиц и агрегаты для трендов — см. `sql/schema/*` и `docs/TODO_DATABASE.md`
- Feature Store (Redis/DB): `src/utils/feature_store.py`

## Безопасность

- JWT, аудит сессий, базовая RBAC, HTTPS в продакшене (Nginx)

## Развертывание

Разработка:

```bash
docker compose -f docker-compose.dev.yml up -d
```

Прод/стейджинг (пример):

```bash
docker compose -f docker-compose.yml up -d
```

## Документация

- [API документация](docs/api/README.md)
- [Руководство по развертыванию](docs/deployment/README.md)
- [Руководство пользователя](docs/user_guide/README.md)

## Поддержка

При возникновении проблем:

1. Проверьте логи: `make logs`
2. Проверьте статус: `make status`
3. Проверьте здоровье: `make health`


## Команда

Цифровой Синтез
