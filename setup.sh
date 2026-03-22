#!/bin/bash
# Setup script for LFM2.5-Audio WebSocket Server on RunPod

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install websockets
pip install transformers accelerate
pip install liquid-audio --ignore-requires-python

echo "=== Downloading model (this takes a few minutes) ==="
python -c "
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
print('Downloading LFM2.5-Audio model...')
LFM2AudioProcessor.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B')
LFM2AudioModel.from_pretrained('LiquidAI/LFM2.5-Audio-1.5B')
print('Model downloaded!')
"

echo "=== Setup complete! ==="
echo "Run the server with: python websocket_server.py"
