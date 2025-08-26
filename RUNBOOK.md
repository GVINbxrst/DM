# Руководство по запуску проекта (Ubuntu VM и локально Windows/Linux)

Этот документ содержит проверенные шаги для запуска DiagnosticModel:
- На виртуальной машине Ubuntu (через Docker Compose)
- Локально без Docker (Windows/Linux) для разработки и отладки

Сервисы и порты по умолчанию:
- API (FastAPI): http://localhost:8000 (документация: /docs)
- Dashboard (Streamlit): http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (логин/пароль по умолчанию: admin/admin123)
- PostgreSQL: localhost:5432
- Redis: localhost:6379

---

## 1) Запуск на виртуальной машине Ubuntu (Docker Compose)

Рекомендации по VM: Ubuntu 22.04 LTS+, 2–4 vCPU, 4–8 GB RAM, 20+ GB SSD.

### 1.1 Установка зависимостей

```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git

# Docker Engine + Compose plugin
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Разрешить текущему пользователю работать с docker без sudo
sudo usermod -aG docker $USER
# Перелогиньтесь (или reboot), чтобы группа вступила в силу
```

Проверка:

```bash
docker --version
docker compose version
```

### 1.2 Клонирование репозитория

```bash
git clone <https://github.com/GVINbxrst/DM>
cd DiagnosticModel
```

(Опционально) Создайте и отредактируйте `.env` при необходимости. Для dev-стека большинство переменных уже зашито в compose-файл.

### 1.3 Запуск DEV‑стека (hot‑reload)

```bash
# Поднимаем инфраструктуру
docker compose -f docker-compose.dev.yml up -d postgres redis

# Поднимаем приложение и мониторинг
docker compose -f docker-compose.dev.yml up -d api worker dashboard prometheus grafana

# (опционально) Jupyter
# docker compose -f docker-compose.dev.yml up -d jupyter
```

Проверки:
- API: curl http://localhost:8000/health
- Dashboard: откройте http://<VM_IP>:8501
- Grafana: http://<VM_IP>:3000 (admin/admin123)

Применение Alembic миграций (если требуется):

```bash
docker compose -f docker-compose.dev.yml exec api alembic upgrade head
```

### 1.4 Загрузка CSV (loader)

В `docker-compose.dev.yml` есть сервис `loader`, монтирующий локальную папку `./data` в контейнер `/data`.

```bash
# Поместите файлы CSV в папку проекта ./data
# Формат CSV строгий: первая строка-заголовок "current_R,current_S,current_T"

# Загрузка одного файла
docker compose -f docker-compose.dev.yml run --rm loader \
  python src/data_processing/csv_loader.py /data/sample.csv --sample-rate 25600

# Загрузка всех CSV из каталога
docker compose -f docker-compose.dev.yml run --rm loader \
  python src/data_processing/csv_loader.py /data --sample-rate 25600 --batch-size 10000
```

### 1.5 Управление и отладка

```bash
# Логи сервиса
docker compose -f docker-compose.dev.yml logs -f api

# Перезапуск сервиса
docker compose -f docker-compose.dev.yml restart api

# Остановка и очистка (осторожно: удаляет volumes при -v)
docker compose -f docker-compose.dev.yml down
# docker compose -f docker-compose.dev.yml down -v  # с удалением томов
```

### 1.6 Прод‑ориентированный запуск (опционально)

```bash
# Сборка и запуск по docker-compose.yml
make docker-build
make docker-up

# Или вручную:
# docker compose up -d
```

---

## 2) Локальный запуск без Docker (Windows / Linux)

Подходит для отладки и быстрого E2E без Redis (eager‑режим Celery).

### 2.1 Предварительные требования
- Python 3.11+ (x64)
- Git (опционально)
- Redis (опционально, если не используется eager‑режим)

### 2.2 Создание виртуального окружения и установка зависимостей

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# Для разработки дополнительно: pip install -r requirements-dev.txt
```

Linux bash:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Для разработки: pip install -r requirements-dev.txt
```

### 2.3 Конфигурация (.env)

Минимальный пример для SQLite:
```dotenv
APP_ENVIRONMENT=development
DATABASE_URL=sqlite+aiosqlite:///./local.db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
LOG_LEVEL=INFO
```

Для локального PostgreSQL:
```dotenv
APP_ENVIRONMENT=development
DATABASE_URL=postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
LOG_LEVEL=INFO
```

### 2.4 Применение миграций Alembic

Windows PowerShell:
```powershell
# вариант 1: прочитать значение из .env
$env:DATABASE_URL=(Get-Content .env | Select-String '^DATABASE_URL=').ToString().Split('=')[1]
alembic upgrade head

# вариант 2: задать вручную
$env:DATABASE_URL="sqlite+aiosqlite:///./local.db"
alembic upgrade head
```

Linux bash:
```bash
export DATABASE_URL="sqlite+aiosqlite:///./local.db"  # или ваш postgres URL
alembic upgrade head
```

Примечание: для SQLite часть миграций может быть PostgreSQL‑специфичной; в таком случае допустимо запускаться без alembic, таблицы будут созданы при первом подключении ORM (см. src/database/connection.py).

### 2.5 Запуск API (Uvicorn)

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Linux bash:
```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:
- http://localhost:8000/health
- http://localhost:8000/docs

### 2.6 Redis и Celery worker (или eager‑режим)

Вариант A. С Redis:

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
celery -A src.worker.config:celery_app worker --loglevel=info --concurrency=2
```

Linux bash:
```bash
source .venv/bin/activate
celery -A src.worker.config:celery_app worker --loglevel=info --concurrency=2
```

Вариант B. Без Redis (eager‑режим):

Windows PowerShell:
```powershell
$env:CELERY_TASK_ALWAYS_EAGER="1"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Linux bash:
```bash
export CELERY_TASK_ALWAYS_EAGER=1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2.7 Загрузка CSV и проверка конвейера

Формат CSV: заголовок первой строки — `current_R,current_S,current_T`. Пустые значения для S/T допустимы.

Через API (Windows PowerShell):
```powershell
Invoke-WebRequest `
  -Uri http://localhost:8000/api/v1/upload `
  -Method POST `
  -Form @{ file = Get-Item .\sample.csv }
```

Через loader в Docker (даже локально): см. раздел 1.4 — можно подключиться только к postgres/redis из compose.

Проверка таблиц (PostgreSQL):
```powershell
psql "postgresql://diagmod_user:diagmod_password@localhost:5432/diagmod" -c "SELECT COUNT(*) FROM raw_signals;"
psql "postgresql://diagmod_user:diagmod_password@localhost:5432/diagmod" -c "SELECT COUNT(*) FROM features;"
```

### 2.8 Полезные цели Makefile (локально)

```bash
make test           # pytest с покрытием
make format         # black + isort
make lint           # flake8 + mypy
make process-raw    # обработка необработанных RawSignal (eager)
make anomalies      # детекция аномалий (eager)
make forecast       # краткосрочный прогноз (eager)
make report         # сводный отчёт
```

---

## 3) Частые проблемы и решения

- Docker: «permission denied» — добавьте пользователя в группу docker и перелогиньтесь (`sudo usermod -aG docker $USER`).
- Порт занят (8000/8501/9090) — измените порты в compose или остановите конфликтующие службы.
- Alembic не видит DATABASE_URL — убедитесь, что переменная окружения выставлена в текущей сессии перед запуском.
- Redis не установлен — временно используйте eager‑режим: `CELERY_TASK_ALWAYS_EAGER=1`.
- Дубликаты CSV — у `raw_signals.file_hash` уникальный индекс; повторы будут пропущены.

---

## 4) Ссылки

- docker-compose.dev.yml — dev‑стек (api/worker/dashboard/prometheus/grafana/jupyter/loader)
- docker-compose.yml — prod‑ориентированный стек
- src/data_processing/csv_loader.py — правила формата CSV и загрузка
- src/utils/feature_store.py — Feature Store (db/redis)
- docs/TODO_DATABASE.md — заметки по миграциям и партиционированию
