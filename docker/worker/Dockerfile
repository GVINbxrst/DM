# syntax=docker/dockerfile:1.6
# =============================================================================
# Multi-stage Dockerfile for DiagMod Celery Worker
# =============================================================================

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    C_FORCE_ROOT=1

# Install system dependencies for ML and signal processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libgomp1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-base.txt ./
COPY requirements-ml.txt ./
COPY requirements.txt ./
COPY requirements-dev.txt ./

# Development stage
FROM base as development

# Install development dependencies with pip cache
ARG INSTALL_DEV=0
ARG LIGHT_DEPS=0
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c "\
    python -m pip install --upgrade pip setuptools wheel && \
    pip wheel -r requirements-base.txt -w /app/.wheelhouse || true; \
    if [ \"$LIGHT_DEPS\" != \"1\" ]; then pip wheel -r requirements-ml.txt -w /app/.wheelhouse || true; fi; \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-base.txt && \
    if [ \"$LIGHT_DEPS\" != \"1\" ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-ml.txt; fi && \
    if [ -f requirements-dev.txt ] && [ \"$INSTALL_DEV\" = \"1\" ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-dev.txt; fi"

# Copy only necessary code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

# Switch to app user
USER app

# Command for development
CMD ["python", "-m", "celery", "-A", "src.worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

# Production stage
FROM base as production

# Install only production dependencies with pip cache
ARG LIGHT_DEPS=0
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c "\
    python -m pip install --upgrade pip setuptools wheel && \
    pip wheel -r requirements-base.txt -w /app/.wheelhouse || true; \
    if [ \"$LIGHT_DEPS\" != \"1\" ]; then pip wheel -r requirements-ml.txt -w /app/.wheelhouse || true; fi; \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-base.txt && \
    if [ \"$LIGHT_DEPS\" != \"1\" ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-ml.txt; fi"

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/.cache && \
    chown -R app:app /app

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m celery -A src.worker.celery_app inspect ping || exit 1

# Command for production
CMD ["python", "-m", "celery", "-A", "src.worker.celery_app", "worker", \
    "--loglevel=info", \
    "--concurrency=4", \
    "--prefetch-multiplier=1", \
    "--max-tasks-per-child=1000", \
    "--time-limit=7200", \
    "--soft-time-limit=3600"]
