# Основные локальные ссылки (Dev)

Ниже — быстрые ссылки для локального окружения docker-compose.dev:

- API (FastAPI)
  - http://localhost:8000/
  - http://localhost:8000/health
  - http://localhost:8000/monitoring/metrics
  - http://localhost:8000/docs (Swagger)
  - http://localhost:8000/redoc (ReDoc)

- Dashboard (Streamlit)
  - http://localhost:8501/
  - http://localhost:8501/healthz

- Prometheus
  - http://localhost:9090/
  - http://localhost:9090/targets
  - http://localhost:9090/graph

- Grafana
  - http://localhost:3000/login (логин: admin, пароль: admin123)
  - http://localhost:3000/

- Jupyter
  - http://localhost:8888/ (нужен токен из логов контейнера)

Примечание: Postgres (5432) и Redis (6379) — TCP-порты, не HTTP.
