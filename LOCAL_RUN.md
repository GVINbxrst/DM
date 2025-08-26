# Локальный запуск без Docker (Windows / Python 3.11+)

Этот гайд объясняет, как запустить DiagnosticModel локально без Docker: API (FastAPI), база данных (SQLite или PostgreSQL), Celery worker и проверка пайплайна загрузки CSV.

## Предварительные требования

- Python 3.11 или 3.12 (x64)
- Windows PowerShell (v5.1+) или PowerShell 7
- Git (опционально)

Репозиторий предполагает работу из корня проекта `e:\Prj\Dabl\DiagnosticModel`.

## 1) Подготовка окружения (venv)

1. Создать и активировать виртуальное окружение:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   $env:VIRTUAL_ENV  # проверка, что окружение активно
   ```

2. Установить зависимости. В проекте есть и `pyproject.toml`, и `requirements.txt` — это осознанно:
   - `pyproject.toml` — декларативный манифест для сборки/публикации пакета (совместим с pip/uv).
   - `requirements.txt` — «замороженный» список для воспроизводимого окружения разработчика.

   Для локального запуска используйте requirements.txt, чтобы избежать несовместимостей версий:

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   # Для разработки дополнительно:
   # pip install -r requirements-dev.txt
   ```

Примечание: добавлен пакет `aiosqlite` для работы с SQLite через SQLAlchemy Async.

## 2) Конфигурация окружения (.env)

Создайте файл `.env` в корне проекта. Ниже два готовых варианта DATABASE_URL.

A) Быстрый старт на SQLite:

```env
APP_ENVIRONMENT=development
DATABASE_URL=sqlite+aiosqlite:///./local.db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
LOG_LEVEL=INFO
```

B) Локальный PostgreSQL:

1. Установите PostgreSQL (Windows installer) и создайте пользователя/БД:

   ```powershell
   psql -U postgres -c "CREATE ROLE diagmod_user WITH LOGIN PASSWORD 'diagmod_password';"
   psql -U postgres -c "CREATE DATABASE diagmod OWNER diagmod_user;"
   psql -U postgres -d diagmod -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
   ```

2. В `.env` укажите:

```env
APP_ENVIRONMENT=development
DATABASE_URL=postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
LOG_LEVEL=INFO
```

Важно: Alembic использует переменную окружения `DATABASE_URL`. Для SQLite она должна совпадать с форматом `sqlite+aiosqlite:///./local.db`.

## 3) Применение миграций Alembic

Активируйте venv и выполните:

```powershell
$env:DATABASE_URL=(Get-Content .env | Select-String '^DATABASE_URL=').ToString().Split('=')[1]
alembic upgrade head
```

Если .env не читается так, можно выставить вручную:

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///./local.db"  # или ваш postgres URL
alembic upgrade head
```

SQLite: таблицы создаются автоматически при первом подключении (см. `src/database/connection.py`).
Миграции Alembic для SQLite можно пропустить, так как часть миграций содержит PostgreSQL-специфичные типы
и может не примениться. Уникальные индексы создаются через SQLAlchemy-модель (см. `models.RawSignal.__table_args__`).

## 4) Запуск API

В одном окне PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:
- http://localhost:8000/ — должен вернуть `{status: healthy}`
- http://localhost:8000/docs — Swagger UI

## 5) Запуск Redis и Celery

Нужен запущенный Redis (локально установленный сервис или, временно, Docker-only). Если Redis не установлен, можно запустить Celery в «eager» режиме (без брокера), установив переменную `CELERY_TASK_ALWAYS_EAGER=1`.

Вариант 1. С брокером Redis:

```powershell
.\.venv\Scripts\Activate.ps1
celery -A src.worker.config:celery_app worker --loglevel=info --concurrency=2
```

Вариант 2. Без брокера (eager-режим):

```powershell
.\.venv\Scripts\Activate.ps1
$env:CELERY_TASK_ALWAYS_EAGER="1"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

В eager-режиме загрузка CSV сразу выполняет пайплайн синхронно (работает для быстрой проверки без Redis).

## 6) Формат CSV и загрузка

Ожидается строгий формат:
- Заголовок: одна колонка со строкой `current_R,current_S,current_T`
- Каждая последующая строка: три значения через запятую, пустые допустимы (пустое -> NaN)

Пример файла:

```
current_R,current_S,current_T
0.1,0.2,0.3
0.0,,0.1
,,
0.2,0.2,0.2
```

Загрузка в API:

```powershell
Invoke-WebRequest `
  -Uri http://localhost:8000/api/v1/upload `
  -Method POST `
  -Form @{ file = Get-Item .\sample.csv }
```

Ответ должен содержать `raw_id` и `status` (queued|direct-*).

## 7) Проверка пайплайна

1. Убедиться, что API запущен. Если используете Redis — запустите Celery worker.
2. Загрузить тестовый CSV как выше.
3. Проверить, что запись появилась:

   - В `raw_signals` (поле `file_hash` уникально)
   - После обработки должны появиться записи в `features`

Проверка через SQLite CLI (пример):

```powershell
# Пример для SQLite3 (если установлен sqlite3.exe)
sqlite3.exe .\local.db "SELECT COUNT(*) FROM raw_signals;"
sqlite3.exe .\local.db "SELECT COUNT(*) FROM features;"
```

Для PostgreSQL используйте psql:

```powershell
psql "postgresql://diagmod_user:diagmod_password@localhost:5432/diagmod" -c "SELECT COUNT(*) FROM raw_signals;"
psql "postgresql://diagmod_user:diagmod_password@localhost:5432/diagmod" -c "SELECT COUNT(*) FROM features;"
```

## 8) Частые вопросы

- Дубли зависимостей: pyproject и requirements.
  - Для локального запуска используйте requirements.txt (забитые версии). pyproject полезен для сборки и может иметь более широкие версии; поэтому во избежание конфликтов мы придерживаемся requirements в dev.
  - Если используете `pip install .` из pyproject, убедитесь, что вручную доустановлен `aiosqlite` для SQLite.

- DATABASE_URL не подхватывается Alembic.
  - Убедитесь, что переменная окружения установлена перед `alembic upgrade head`.

- Celery не видит брокер.
  - Проверьте `CELERY_BROKER_URL`, что Redis запущен и доступен. Временно можно включить eager-режим: `$env:CELERY_TASK_ALWAYS_EAGER="1"`.

Готово. Теперь можно загружать CSV и получать аналитику локально.
