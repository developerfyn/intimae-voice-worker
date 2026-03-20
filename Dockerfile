# RunPod Worker for LFM2.5-Audio (Liquid AI)
# Speech-to-Speech AI model

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools for audio packages)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install runpod first
RUN pip install --no-cache-dir runpod

# Install audio processing dependencies
RUN pip install --no-cache-dir torchaudio

# Install transformers and accelerate
RUN pip install --no-cache-dir transformers accelerate

# Install liquid-audio (may need additional deps)
RUN pip install --no-cache-dir liquid-audio

# Optional: Install flash attention for better performance
RUN pip install flash-attn --no-build-isolation || echo "Flash attention not available, using SDPA fallback"

# Download model during build (baked into image for faster cold starts)
RUN python -c "from liquid_audio import LFM2AudioModel, LFM2AudioProcessor; \
    LFM2AudioProcessor.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B'); \
    LFM2AudioModel.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B')"

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
