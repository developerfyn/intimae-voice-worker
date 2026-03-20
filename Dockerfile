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

# Install liquid-audio (ignore Python version check)
RUN pip install --no-cache-dir liquid-audio --ignore-requires-python

# Optional: Install flash attention for better performance
RUN pip install flash-attn --no-build-isolation || echo "Flash attention not available"

# Download model during build (baked into image for faster cold starts)
RUN python -c "from liquid_audio import LFM2AudioModel, LFM2AudioProcessor; \
    print('Downloading model...'); \
    LFM2AudioProcessor.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B'); \
    LFM2AudioModel.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B'); \
    print('Model downloaded successfully')"

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
