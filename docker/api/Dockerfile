# syntax=docker/dockerfile:1.6
# =============================================================================
# Multi-stage Dockerfile for DiagMod API Service
# =============================================================================

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching (avoid copying pyproject here)
COPY requirements.txt ./
COPY requirements-dev.txt ./

# Development stage
FROM base as development

# Install development dependencies with pip cache (BuildKit)
# Optional: also install dev-only tools (jupyter) when INSTALL_DEV/INSTALL_JUPYTER are set
ARG INSTALL_DEV=0
ARG INSTALL_JUPYTER=0
# When LIGHT_JUPYTER=1 we skip heavy prod requirements and install only a light Jupyter stack
ARG LIGHT_JUPYTER=0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$LIGHT_JUPYTER" = "1" ]; then \
        PIP_NO_CACHE_DIR=0 pip install --upgrade pip setuptools wheel && \
        PIP_NO_CACHE_DIR=0 pip install jupyterlab ipykernel numpy pandas matplotlib seaborn; \
        if [ "$INSTALL_DEV" = "1" ] && [ -f requirements-dev.txt ]; then \
            PIP_NO_CACHE_DIR=0 pip install -r requirements-dev.txt; \
        fi; \
    else \
        PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=5 PIP_NO_CACHE_DIR=0 pip install -r requirements.txt && \
        if [ "$INSTALL_DEV" = "1" ] && [ -f requirements-dev.txt ]; then \
            PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=5 PIP_NO_CACHE_DIR=0 pip install -r requirements-dev.txt; \
        fi && \
        if [ "$INSTALL_JUPYTER" = "1" ]; then \
            PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=5 PIP_NO_CACHE_DIR=0 pip install jupyterlab; \
        fi; \
    fi

# Copy only sources needed for dev runtime
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

# Switch to app user
USER app

# Expose ports
EXPOSE 8000 8001

# Command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install -r requirements.txt

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/.cache && \
    chown -R app:app /app

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Command for production
CMD ["gunicorn", "src.api.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
