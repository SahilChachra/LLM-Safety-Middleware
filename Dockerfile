# ─────────────────────────────────────────────────────────────────────────────
# LLM Safety Middleware — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Upgrade to 3.11-slim for security patches and speed.
# Build: docker build -t llm-safety-pipeline .
# Run:   docker run -p 8000:8000 --env-file .env llm-safety-pipeline
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Prevents .pyc files and buffered stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Default config path (override with -e CONFIG_PATH=...)
    CONFIG_PATH=config_production.json

WORKDIR /app

# System dependencies: only curl (needed for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer → cached on reruns)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy NLP model
RUN python -m spacy download en_core_web_sm

# Copy application source and documentation
COPY llm_safety_pipeline.py .
COPY api_server.py .
COPY demo_safety_pipeline.py .
COPY config_production.json .
COPY README.md .

# Create runtime directories with correct permissions
RUN mkdir -p /app/logs /app/safety_reports && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "api_server.py"]
