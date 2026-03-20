# LFM2.5-Audio RunPod Worker

Serverless worker for Liquid AI's LFM2.5-Audio speech-to-speech model.

## Features

- **Text-to-Speech (TTS)**: Convert text to natural speech
- **Speech-to-Speech (S2S)**: Process audio input, generate audio response
- **Context Sync**: Inject text conversation history for seamless text↔voice switching
- **Voice Selection**: Multiple voice styles available

## Build & Deploy

### 1. Build Docker Image

```bash
cd runpod-lfm
docker build -t lfm-audio-worker .
```

### 2. Push to Docker Hub

```bash
docker tag lfm-audio-worker YOUR_DOCKERHUB_USERNAME/lfm-audio-worker:latest
docker push YOUR_DOCKERHUB_USERNAME/lfm-audio-worker:latest
```

### 3. Deploy on RunPod

Create a serverless endpoint using the image:
- Image: `YOUR_DOCKERHUB_USERNAME/lfm-audio-worker:latest`
- GPU: RTX 4090 (24GB VRAM)
- Min Workers: 0
- Max Workers: 3

## API Usage

### Text-to-Speech

```json
{
  "input": {
    "text": "Hello! How are you today?",
    "voice": "UK female",
    "system_prompt": "You are a warm, friendly AI companion."
  }
}
```

### Speech-to-Speech

```json
{
  "input": {
    "audio": "<base64 encoded WAV/MP3>",
    "system_prompt": "You are a caring AI companion named Luna.",
    "voice": "US female"
  }
}
```

### With Conversation History (Context Sync)

```json
{
  "input": {
    "audio": "<base64 audio>",
    "system_prompt": "You are Luna, a caring AI companion.",
    "voice": "UK female",
    "history": [
      {"role": "user", "content": "I had a rough day at work."},
      {"role": "assistant", "content": "I'm sorry to hear that. Want to talk about it?"},
      {"role": "user", "content": "My boss was really critical of my presentation."}
    ]
  }
}
```

## Response Format

```json
{
  "audio": "<base64 WAV>",
  "transcript": "I understand how difficult that must feel...",
  "sample_rate": 24000,
  "format": "wav"
}
```

## Voice Options

- `UK male` / `UK female`
- `US male` / `US female`
- Custom voice descriptions work too (e.g., "warm, gentle female voice")

## Cost Estimate

- RTX 4090 Serverless: ~$0.77/hr
- Typical inference: 2-5 seconds per response
- Estimated cost: ~$0.001-0.003 per voice exchange
