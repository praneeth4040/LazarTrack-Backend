FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV, FFmpeg, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Stream logs immediately (no Python buffering)
ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT at runtime (default 8080)
ENV PORT=8080
EXPOSE 8080

# Single worker — Cloud Run scales by spinning new containers, not internal workers.
# timeout-keep-alive 620 gives requests the full 600s before the connection closes.
CMD ["sh", "-c", \
     "python -m uvicorn app.main:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers 1 \
      --timeout-keep-alive 620"]
