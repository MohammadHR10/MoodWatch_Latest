FROM python:3.10-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    TORCH_HOME=/app/torch_cache \
    HF_HOME=/app/hf_cache

# System deps - minimal for lightweight detector
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY lightweight_detector.py .

# Pre-download models during build for faster cold starts
RUN python -c "from lightweight_detector import get_detector; d = get_detector(); print('Lightweight models ready')" || echo "Model preload skipped"

EXPOSE 8080

# Faster health check since we're lighter now
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Can use more workers now since we're lighter
CMD exec gunicorn --bind :$PORT --workers 2 --threads 2 --timeout 120 app:app
