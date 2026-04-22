FROM python:3.10-slim

WORKDIR /app

# System libs for OpenCV headless + MediaPipe + chromadb C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1-mesa-glx \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

# Core requirements (no PyTorch / YOLO — too slow on CPU; disabled via try/except)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces uses 7860; local default is 8000
ENV PORT=7860
EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
