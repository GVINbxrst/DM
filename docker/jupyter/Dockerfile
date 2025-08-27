# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-base.txt ./
COPY requirements-ml.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c "python -m pip install --upgrade pip setuptools wheel && \
    pip wheel -r requirements-base.txt -w /app/.wheelhouse || true; \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-base.txt && \
    pip install jupyterlab ipykernel && \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-ml.txt || true"

COPY --chown=root:root src/ ./src/
COPY --chown=root:root pyproject.toml ./

EXPOSE 8888
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
