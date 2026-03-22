"""
Real-time Voice Chat WebSocket Server for LFM2.5-Audio
Bidirectional audio streaming like a phone call
"""

import asyncio
import websockets
import torch
import torchaudio
import json
import base64
import io
import os
from typing import Optional

# Global model instances
processor = None
model = None
HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"


def load_model():
    """Load model once when server starts."""
    global processor, model

    if processor is None:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        print("Loading LFM2.5-Audio model...")
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO)
        model = LFM2AudioModel.from_pretrained(HF_REPO)

        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("Warning: Running on CPU (will be slow)")

    return processor, model


def decode_audio(audio_base64: str) -> torch.Tensor:
    """Decode base64 audio to tensor."""
    audio_bytes = base64.b64decode(audio_base64)
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)

    # Resample to 24kHz if needed
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)

    return waveform


def encode_audio_chunk(waveform: torch.Tensor) -> str:
    """Encode audio tensor to base64 WAV."""
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform.cpu(), 24000, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


class VoiceSession:
    """Manages a single voice chat session."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.processor, self.model = load_model()
        self.system_prompt = "You are a warm, friendly AI companion."
        self.voice = "UK female"
        self.history = []
        self.is_active = True

    async def handle_message(self, message: dict):
        """Handle incoming WebSocket message."""
        msg_type = message.get("type")

        if msg_type == "config":
            # Configure session
            self.system_prompt = message.get("system_prompt", self.system_prompt)
            self.voice = message.get("voice", self.voice)
            self.history = message.get("history", [])
            await self.send({"type": "config_ack", "status": "ok"})

        elif msg_type == "audio":
            # Process audio and stream response
            audio_base64 = message.get("audio")
            if audio_base64:
                await self.process_audio(audio_base64)

        elif msg_type == "text":
            # TTS mode - convert text to speech
            text = message.get("text")
            if text:
                await self.process_text(text)

        elif msg_type == "ping":
            await self.send({"type": "pong"})

        elif msg_type == "end":
            self.is_active = False
            await self.send({"type": "session_ended"})

    async def process_audio(self, audio_base64: str):
        """Process audio input and stream audio response."""
        from liquid_audio import ChatState

        try:
            # Decode input audio
            waveform = decode_audio(audio_base64)
            if torch.cuda.is_available():
                waveform = waveform.cuda()

            # Build chat context
            chat = ChatState(self.processor)

            # System prompt
            chat.new_turn("system")
            chat.add_text(f"{self.system_prompt} Use the {self.voice} voice.")
            chat.end_turn()

            # Add conversation history
            for msg in self.history:
                chat.new_turn(msg["role"])
                chat.add_text(msg["content"])
                chat.end_turn()

            # Add user audio
            chat.new_turn("user")
            audio_tokens = self.processor.encode(waveform)
            chat.add_audio(audio_tokens)
            chat.end_turn()

            # Start assistant response
            chat.new_turn("assistant")

            # Signal start of response
            await self.send({"type": "response_start"})

            # Generate and stream response
            audio_chunks = []
            text_tokens = []
            chunk_size = 4800  # 200ms at 24kHz
            current_chunk = []

            for t in self.model.generate_sequential(
                **chat.to_model_inputs(),
                max_new_tokens=2048
            ):
                if t.numel() > 1:
                    # Audio token
                    current_chunk.append(t)

                    # Stream chunk when we have enough
                    if len(current_chunk) >= 10:
                        audio_codes = torch.stack(current_chunk, 1).unsqueeze(0)
                        chunk_waveform = self.processor.decode(audio_codes)
                        chunk_b64 = encode_audio_chunk(chunk_waveform)

                        await self.send({
                            "type": "audio_chunk",
                            "audio": chunk_b64,
                            "sample_rate": 24000
                        })

                        audio_chunks.extend(current_chunk)
                        current_chunk = []
                else:
                    text_tokens.append(t)

            # Send remaining audio
            if current_chunk:
                audio_codes = torch.stack(current_chunk, 1).unsqueeze(0)
                chunk_waveform = self.processor.decode(audio_codes)
                chunk_b64 = encode_audio_chunk(chunk_waveform)

                await self.send({
                    "type": "audio_chunk",
                    "audio": chunk_b64,
                    "sample_rate": 24000
                })

            # Get transcript
            transcript = self.processor.decode_text(text_tokens) if text_tokens else None

            # Signal end of response
            await self.send({
                "type": "response_end",
                "transcript": transcript
            })

        except Exception as e:
            await self.send({
                "type": "error",
                "message": str(e)
            })

    async def process_text(self, text: str):
        """Convert text to speech (TTS mode)."""
        from liquid_audio import ChatState

        try:
            chat = ChatState(self.processor)

            # System prompt
            chat.new_turn("system")
            chat.add_text(f"Convert the following text to speech. Use the {self.voice} voice.")
            chat.end_turn()

            # User text
            chat.new_turn("user")
            chat.add_text(text)
            chat.end_turn()

            # Generate
            chat.new_turn("assistant")

            await self.send({"type": "response_start"})

            audio_chunks = []
            for t in self.model.generate_sequential(
                **chat.to_model_inputs(),
                max_new_tokens=2048
            ):
                if t.numel() > 1:
                    audio_chunks.append(t)

            if audio_chunks:
                audio_codes = torch.stack(audio_chunks[:-1], 1).unsqueeze(0)
                waveform = self.processor.decode(audio_codes)
                audio_b64 = encode_audio_chunk(waveform)

                await self.send({
                    "type": "audio_chunk",
                    "audio": audio_b64,
                    "sample_rate": 24000
                })

            await self.send({
                "type": "response_end",
                "transcript": text
            })

        except Exception as e:
            await self.send({
                "type": "error",
                "message": str(e)
            })

    async def send(self, message: dict):
        """Send JSON message to client."""
        await self.websocket.send(json.dumps(message))


async def handle_connection(websocket, path):
    """Handle a new WebSocket connection."""
    print(f"New connection from {websocket.remote_address}")
    session = VoiceSession(websocket)

    try:
        # Send ready signal
        await session.send({
            "type": "ready",
            "model": HF_REPO,
            "sample_rate": 24000
        })

        async for message in websocket:
            try:
                data = json.loads(message)
                await session.handle_message(data)

                if not session.is_active:
                    break

            except json.JSONDecodeError:
                await session.send({
                    "type": "error",
                    "message": "Invalid JSON"
                })

    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed: {websocket.remote_address}")

    finally:
        print(f"Session ended: {websocket.remote_address}")


async def main():
    """Start the WebSocket server."""
    # Load model at startup
    print("Initializing server...")
    load_model()

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8765))

    print(f"Starting WebSocket server on ws://{host}:{port}")

    async with websockets.serve(handle_connection, host, port):
        print("Server ready! Waiting for connections...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
