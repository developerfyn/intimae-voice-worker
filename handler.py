"""
RunPod Serverless Handler for LFM2.5-Audio
Speech-to-Speech AI model by Liquid AI
"""

import runpod
import torch
import torchaudio
import base64
import io
import os
from typing import Optional

# Global model instances (loaded once per worker)
processor = None
model = None

HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"


def load_model():
    """Load model once when worker starts."""
    global processor, model

    if processor is None:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        print("Loading LFM2.5-Audio model...")
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("Warning: Running on CPU (slow)")

    return processor, model


def decode_audio(audio_base64: str) -> torch.Tensor:
    """Decode base64 audio to tensor."""
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)

    # Resample to 24kHz if needed (LFM2.5 uses 24kHz)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)

    return waveform


def encode_audio(waveform: torch.Tensor) -> str:
    """Encode audio tensor to base64 WAV."""
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform.cpu(), 24000, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for LFM2.5-Audio.

    Input modes:
    1. Text-to-Speech (TTS):
       {"text": "Hello world", "voice": "UK male"}

    2. Speech-to-Speech (S2S):
       {"audio": "<base64 audio>", "system_prompt": "Respond warmly"}

    3. Text + Audio context:
       {"text": "Continue the story", "context_audio": "<base64>", "system_prompt": "..."}
    """
    from liquid_audio import ChatState

    load_model()
    job_input = job.get("input", {})

    # Extract inputs
    text_input = job_input.get("text")
    audio_input = job_input.get("audio")  # Base64 encoded audio
    system_prompt = job_input.get("system_prompt", "You are a helpful AI assistant.")
    voice = job_input.get("voice", "UK female")  # Voice style
    max_tokens = job_input.get("max_tokens", 1024)

    # Conversation history (for context sync with text chat)
    history = job_input.get("history", [])  # [{"role": "user", "content": "..."}, ...]

    try:
        chat = ChatState(processor)

        # System prompt
        chat.new_turn("system")
        chat.add_text(f"{system_prompt} Use the {voice} voice.")
        chat.end_turn()

        # Add conversation history if provided (for context sync)
        for msg in history:
            chat.new_turn(msg["role"])
            chat.add_text(msg["content"])
            chat.end_turn()

        # Add current input
        chat.new_turn("user")

        if audio_input:
            # Speech-to-Speech: process audio input
            waveform = decode_audio(audio_input)
            if torch.cuda.is_available():
                waveform = waveform.cuda()
            audio_tokens = processor.encode(waveform)
            chat.add_audio(audio_tokens)
        elif text_input:
            # Text-to-Speech
            chat.add_text(text_input)
        else:
            return {"error": "No input provided. Use 'text' or 'audio' field."}

        chat.end_turn()
        chat.new_turn("assistant")

        # Generate response
        audio_out = []
        text_out = []

        for t in model.generate_sequential(**chat.to_model_inputs(), max_new_tokens=max_tokens):
            if t.numel() > 1:
                audio_out.append(t)
            else:
                # Text token
                text_out.append(t)

        # Decode audio
        if audio_out:
            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            waveform = processor.decode(audio_codes)
            output_audio = encode_audio(waveform)
        else:
            output_audio = None

        # Decode text (for transcript)
        transcript = processor.decode_text(text_out) if text_out else None

        return {
            "audio": output_audio,  # Base64 WAV
            "transcript": transcript,  # Text transcript for saving to DB
            "sample_rate": 24000,
            "format": "wav"
        }

    except Exception as e:
        return {"error": str(e)}


# Initialize model on cold start
load_model()

# Start serverless worker
runpod.serverless.start({"handler": handler})
