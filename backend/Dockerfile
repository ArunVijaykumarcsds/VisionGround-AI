# =============================================================
# Locate Anything Assistant — Backend Dockerfile
# =============================================================
# Stage 1: builder  (installs all Python dependencies)
# Stage 2: runtime  (lean image, no build tools)
#
# Build args:
#   CUDA_VERSION   e.g. 12.1.0  (default: CPU-only base)
#   PYTHON_VERSION e.g. 3.11    (default: 3.11)
#
# GPU build example:
#   docker build \
#     --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
#     -t locate-anything-backend .
#
# CPU build (default):
#   docker build -t locate-anything-backend .
# =============================================================

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE} AS builder

# ---------- system build dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        # lmdb build dep
        liblmdb-dev \
        # decord build deps
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for layer caching.
COPY requirements.txt .

# Install Python deps into a prefix that we'll copy to the runtime stage.
# We install torch CPU by default; GPU images supply CUDA-enabled torch.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================
# Stage 2: runtime
# =============================================================
FROM ${BASE_IMAGE} AS runtime

# Minimal runtime system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source.
COPY app/ ./app/

# Create logs directory.
RUN mkdir -p /app/logs

# Create a non-root user for security.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# =============================================================
# Environment defaults (override at runtime with --env or .env)
# =============================================================
ENV MODEL_PATH=nvidia/LocateAnything-3B \
    DEVICE=auto \
    TORCH_DTYPE=bfloat16 \
    GENERATION_MODE=hybrid \
    MAX_NEW_TOKENS=2048 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    MAX_IMAGE_SIZE_MB=20 \
    MAX_IMAGE_DIMENSION=2560 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Model cache directory — mount a volume here to persist downloads.
# docker run -v /local/model/cache:/app/.cache/huggingface ...
VOLUME ["/app/.cache/huggingface"]

EXPOSE 8000

# Health check — polls /health every 30 s.
# Allows 5 minutes start period for model download + load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch Uvicorn.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--no-access-log"]
