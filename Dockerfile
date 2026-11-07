# Audelle backend image.
#
# CPU-only on purpose: the measured target is commodity CPU hardware. The
# PyTorch CPU wheel index avoids pulling a multi-gigabyte CUDA runtime that the
# models never use here.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUDIO_STORAGE_DIR=/data/storage \
    TORCH_HOME=/data/torch \
    HF_HOME=/data/huggingface \
    XDG_CACHE_HOME=/data/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first; the requirements file deliberately uses torch>=2.0 and would
# otherwise resolve to the default CUDA wheel on Linux.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install -r /app/server/requirements.txt

COPY server /app/server
COPY shared /app/shared
COPY scripts /app/scripts

RUN mkdir -p /data/storage /data/torch /data/huggingface /data/cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
