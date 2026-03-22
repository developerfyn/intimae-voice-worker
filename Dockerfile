# RunPod Worker for LFM2.5-Audio (Liquid AI)
# Speech-to-Speech AI model

# Use PyTorch 2.8 image (liquid-audio requires torch >= 2.8.0)
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install runpod
RUN pip install --no-cache-dir runpod

# Install liquid-audio dependencies explicitly first
RUN pip install --no-cache-dir transformers accelerate

# Install liquid-audio (ignore Python version check)
RUN pip install --no-cache-dir liquid-audio --ignore-requires-python

# Note: Flash attention removed - 2+ hour compile time not worth it for initial deployment
# Note: Model download moved to runtime (handler.py) - build was timing out on 3GB download
# Cold starts will be slower (~60s) but build is more reliable

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
