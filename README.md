# Qwen3-TTS VoiceDesign Server

Internal REST API server for `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.

## Setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `POST /v1/tts/voice-design` — `{"text": "...", "language": "Auto", "instruct": "..."}` → returns `audio/wav`
- `GET /health` — server + GPU status
