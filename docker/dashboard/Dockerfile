# syntax=docker/dockerfile:1.6
# =============================================================================
# Multi-stage Dockerfile for DiagMod Streamlit Dashboard
# =============================================================================

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
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

# Copy requirements first for better caching
COPY requirements-base.txt ./
COPY requirements-ml.txt ./
COPY requirements.txt ./
COPY requirements-dev.txt ./

# Development stage
FROM base as development

# Install development dependencies with pip cache (robust against CRLF)
ARG INSTALL_DEV=0
ARG LIGHT_DEPS=0
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c 'python -m pip install --upgrade pip setuptools wheel && \
    pip wheel -r requirements-base.txt -w /app/.wheelhouse || true; \
    if [ "'$'LIGHT_DEPS'" != "1" ]; then pip wheel -r requirements-ml.txt -w /app/.wheelhouse || true; fi; \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-base.txt && \
    if [ "'$'LIGHT_DEPS'" != "1" ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-ml.txt; fi; \
    if [ "'$'INSTALL_DEV'" = "1" ] && [ -f requirements-dev.txt ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-dev.txt; fi; \
    pip install --no-index --find-links=/app/.wheelhouse "streamlit>=1.28.0" "plotly>=5.18.0" streamlit-option-menu==0.3.6 fpdf2 matplotlib'

# Copy only necessary code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

# Switch to app user
USER app

# Expose port
EXPOSE 8501

# Command for development
CMD ["streamlit", "run", "src/dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Install only production dependencies with pip cache (robust against CRLF)
ARG LIGHT_DEPS=0
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c 'python -m pip install --upgrade pip setuptools wheel && \
    pip wheel -r requirements-base.txt -w /app/.wheelhouse || true; \
    if [ "'$'LIGHT_DEPS'" != "1" ]; then pip wheel -r requirements-ml.txt -w /app/.wheelhouse || true; fi; \
    pip install --no-index --find-links=/app/.wheelhouse -r requirements-base.txt && \
    if [ "'$'LIGHT_DEPS'" != "1" ]; then pip install --no-index --find-links=/app/.wheelhouse -r requirements-ml.txt; fi; \
    pip install --no-index --find-links=/app/.wheelhouse "streamlit>=1.28.0" "plotly>=5.18.0" fpdf2 matplotlib'

# Install additional dependencies for Dashboard
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/app/.wheelhouse \
    sh -c 'pip wheel reportlab==4.0.7 streamlit-authenticator==0.2.3 streamlit-option-menu==0.3.6 -w /app/.wheelhouse || true; \
    pip install --no-index --find-links=/app/.wheelhouse reportlab==4.0.7 streamlit-authenticator==0.2.3 streamlit-option-menu==0.3.6'

# Copy source code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app configs/ ./configs/

# Create necessary directories and streamlit config
RUN mkdir -p /app/logs /app/.streamlit && \
    chown -R app:app /app

# Create Streamlit config
RUN echo '[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#FF6B6B"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"\n\
font = "sans serif"' > /app/.streamlit/config.toml

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Expose port
EXPOSE 8501

# Command for production
CMD ["streamlit", "run", "src/dashboard/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
