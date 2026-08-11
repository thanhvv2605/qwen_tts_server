# Qwen3-TTS VoiceDesign Server

Internal REST API server for `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.

## Setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
```

Requires an NVIDIA GPU with CUDA. The first run downloads the ~4.3GB
`Qwen3-TTS-12Hz-1.7B-VoiceDesign` model from HuggingFace — if the server
seems to hang at startup, that's most likely why; use `python
scripts/smoke_test.py` once `/health` reports `model_loaded: true`, not
before.

## Run

Either of these works; explicit `uvicorn` CLI flags always win over `QWEN_TTS_HOST`/`QWEN_TTS_PORT`:

```bash
# CLI flags (ignores QWEN_TTS_HOST / QWEN_TTS_PORT)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# module entrypoint (honors QWEN_TTS_HOST / QWEN_TTS_PORT from the environment)
python -m app.main
```

The server is unauthenticated and binds `0.0.0.0` by default — keep it behind a
firewall or bind `QWEN_TTS_HOST=127.0.0.1` unless you specifically want it
reachable from other machines.

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
- Response: `audio/wav` binary (200), or `422` (invalid input), `504` (queue timeout), `500` (generation error) — errors return `{"detail": "..."}`

### Jobs API (bất đồng bộ, cho lô lớn)

- `POST /v1/jobs` — gửi 1 job chứa tối đa 1000 items, nhận `job_id` ngay (202)
- `GET /v1/jobs/{job_id}` — poll tiến độ per-item
- `GET /v1/jobs/{job_id}/items/{index}/audio` — tải WAV từng item khi xong
- `DELETE /v1/jobs/{job_id}` — hủy job

Chi tiết và curl mẫu: xem `API.md`. Kết quả job bị xóa mỗi lần server
khởi động lại; job không sống sót qua restart (client gửi lại).

#### Lưu ý vận hành (Jobs API)

- **Dung lượng đĩa**: WAV 24kHz PCM16 ≈ 48KB/giây audio. Một job 578 items
  có thể chiếm ~0.5–1.5GB trong `QWEN_TTS_RESULTS_DIR`. Kết quả chỉ được
  giải phóng khi server restart (thư mục bị xóa sạch lúc khởi động) — theo
  dõi dung lượng đĩa nếu chạy nhiều job lớn giữa các lần restart.
- **`DELETE` là hủy (cancel), không phải xóa**: kết quả các item đã xong
  vẫn tải được sau khi hủy; file trên đĩa không bị xóa.
- **Response của `DELETE` là snapshot trước khi cancel có hiệu lực** — poll
  tiếp `GET /v1/jobs/{id}` đến khi `status` chuyển `cancelled`. Luôn tin
  per-item status làm nguồn chính xác (job hủy sát lúc xong vẫn có thể có
  toàn bộ items `done`).
- **Item `failed` retry thế nào**: gửi lại các item failed trong một job
  mới — job cũ không tự retry.

### `GET /health`

Returns `{"status", "model_loaded", "vram_free_gb", "queue_depth"}`.

## Configuration

All settings are environment variables prefixed `QWEN_TTS_` (see `app/config.py` for defaults), e.g.:

```bash
export QWEN_TTS_PORT=8080          # only takes effect with `python -m app.main` — see Run above
export QWEN_TTS_MAX_BATCH_SIZE=8
export QWEN_TTS_MODEL_ID=./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign  # use a local path to skip re-downloading
export QWEN_TTS_MAX_ITEMS_PER_JOB=1000
export QWEN_TTS_RESULTS_DIR=./results   # bị xóa sạch mỗi lần server khởi động
```

## Manual verification

After starting the server, run `python scripts/smoke_test.py` to confirm the model loaded and a real WAV file comes back.
