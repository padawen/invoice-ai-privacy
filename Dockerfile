# Multi-stage build for Invoice AI Privacy
FROM ubuntu:22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=localhost:11434

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hun \
    tesseract-ocr-deu \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    wget \
    ca-certificates \
    supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/invoice_processing /app/models /var/log/supervisor

# Set permissions
RUN chmod +x /app/models/download_models.sh 2>/dev/null || true

# Create supervisor config
RUN echo '[supervisord]' > /etc/supervisor/conf.d/supervisord.conf && \
    echo 'nodaemon=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=root' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:ollama]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=/usr/local/bin/ollama serve' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=root' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autostart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/supervisor/ollama.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/supervisor/ollama.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:flask]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=python3 app.py' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'directory=/app' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=root' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autostart=false' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/var/log/supervisor/flask.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/var/log/supervisor/flask.log' >> /etc/supervisor/conf.d/supervisord.conf

# Create startup script
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo 'echo "Starting Invoice AI Privacy Service..."' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start supervisord in background' >> /app/start.sh && \
    echo 'supervisord -c /etc/supervisor/conf.d/supervisord.conf &' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Wait for Ollama to be ready' >> /app/start.sh && \
    echo 'echo "Waiting for Ollama to start..."' >> /app/start.sh && \
    echo 'for i in {1..30}; do' >> /app/start.sh && \
    echo '  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then' >> /app/start.sh && \
    echo '    echo "Ollama is ready!"' >> /app/start.sh && \
    echo '    break' >> /app/start.sh && \
    echo '  fi' >> /app/start.sh && \
    echo '  echo "Waiting for Ollama... ($i/30)"' >> /app/start.sh && \
    echo '  sleep 2' >> /app/start.sh && \
    echo 'done' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Check if model exists and pull if necessary' >> /app/start.sh && \
    echo 'MODEL_NAME=${OLLAMA_MODEL:-phi3:mini}' >> /app/start.sh && \
    echo 'echo "Checking for model: $MODEL_NAME"' >> /app/start.sh && \
    echo 'if ! /usr/local/bin/ollama list | grep -q "$MODEL_NAME"; then' >> /app/start.sh && \
    echo '  echo "Pulling model: $MODEL_NAME (this may take a while...)"' >> /app/start.sh && \
    echo '  /usr/local/bin/ollama pull "$MODEL_NAME"' >> /app/start.sh && \
    echo '  echo "Model pulled successfully!"' >> /app/start.sh && \
    echo 'else' >> /app/start.sh && \
    echo '  echo "Model $MODEL_NAME already exists"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Start Flask app' >> /app/start.sh && \
    echo 'echo "Starting Flask application..."' >> /app/start.sh && \
    echo 'supervisorctl start flask' >> /app/start.sh && \
    echo '' >> /app/start.sh && \
    echo '# Keep container running' >> /app/start.sh && \
    echo 'echo "Invoice AI Privacy is ready!"' >> /app/start.sh && \
    echo 'echo "Flask API: http://localhost:5000"' >> /app/start.sh && \
    echo 'echo "Ollama API: http://localhost:11434"' >> /app/start.sh && \
    echo 'tail -f /var/log/supervisor/*.log' >> /app/start.sh && \
    chmod +x /app/start.sh

# Expose ports
EXPOSE 5000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the services
CMD ["/app/start.sh"]