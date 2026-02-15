# Lightweight Cloud Run deployment - calls ML Worker for video AU analysis
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    EMOTION_BACKEND=worker

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements-cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application code
COPY flask_app.py .
COPY config.py .
COPY templates/ templates/
COPY static/ static/

# Create necessary directories
RUN mkdir -p uploads sessions

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 flask_app:app
