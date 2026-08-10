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

### `POST /v1/tts/voice-design`

Request body:
```json
{
  "text": "Xin chao, hom nay troi dep qua.",
  "language": "Auto",
  "instruct": "Giong nu tre, vui ve, toc do noi nhanh."
}
```
- `language` one of: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Response: `audio/wav` binary (200), or `422` (invalid input), `504` (queue timeout), `500` (generation error)

### `GET /health`

Returns `{"status", "model_loaded", "vram_free_gb", "queue_depth"}`.

## Configuration

All settings are environment variables prefixed `QWEN_TTS_` (see `app/config.py` for defaults), e.g.:

```bash
export QWEN_TTS_PORT=8080
export QWEN_TTS_MAX_BATCH_SIZE=8
export QWEN_TTS_MODEL_ID=./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign  # use a local path to skip re-downloading
```

## Manual verification

After starting the server, run `python scripts/smoke_test.py` to confirm the model loaded and a real WAV file comes back.
