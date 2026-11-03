FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid 2GB CUDA wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV OMP_NUM_THREADS=2
ENV PORT=7860

# Create non-root user (uid 1000 required by HF Spaces)
RUN useradd -m -u 1000 appuser
USER 1000

# Copy application code
WORKDIR /home/appuser/app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser data.yaml .
COPY --chown=appuser:appuser eval ./eval

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/healthz')" || exit 1

# Run with single uvicorn worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]