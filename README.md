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

### Voices API (đăng ký & quản lý giọng nói)

- `POST /v1/voices` — đăng ký giọng nói mới từ file WAV + text (201, 409 nếu trùng, 422 nếu không hợp lệ)
- `GET /v1/voices` — liệt kê tất cả giọng nói đã đăng ký (200)
- `DELETE /v1/voices/{voice_id}` — xóa giọng nói (200, 404 nếu không tồn tại)

Chi tiết, curl mẫu, và yêu cầu tên giọng (format `[a-z0-9_-]{1,64}`) + độ dài audio (0.5s–60s):
xem phần Voices API trong `API.md`. Giọng nói lưu trữ vĩnh viễn trong `QWEN_TTS_VOICES_DIR`
và sống sót qua restart (không giống job results).

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

Returns `{"status", "model_loaded", "clone_model_loaded", "vram_free_gb", "queue_depth"}`.

## Configuration

All settings are environment variables prefixed `QWEN_TTS_` (see `app/config.py` for defaults), e.g.:

```bash
export QWEN_TTS_PORT=8080          # only takes effect with `python -m app.main` — see Run above
export QWEN_TTS_MAX_BATCH_SIZE=8
export QWEN_TTS_MODEL_ID=./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign  # use a local path to skip re-downloading
export QWEN_TTS_MAX_ITEMS_PER_JOB=1000
export QWEN_TTS_RESULTS_DIR=./results   # bị xóa sạch mỗi lần server khởi động
export QWEN_TTS_CLONE_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base  # mô hình Base cho voice cloning
export QWEN_TTS_VOICES_DIR=./voices     # thư mục lưu giọng nói (sống sót qua restart)
export QWEN_TTS_VOICE_CLONE_ENABLED=true  # bật/tắt tính năng voice cloning (default: true)
export QWEN_TTS_VOICE_DESIGN_ENABLED=true # tắt để chạy chỉ-clone, tiết kiệm ~4-5GB VRAM (default: true)
```

## Operational notes (Voice Cloning)

When `QWEN_TTS_VOICE_CLONE_ENABLED=true` (default):

- **Dual checkpoints**: server loads both VoiceDesign (~4.3GB) and Base (~4GB) models → ~9–10GB VRAM total.
  First startup downloads the Base model (~4GB) from HuggingFace. `/health` is unreachable while lifespan is
  still loading models, so it can't be used to watch progress — watch the server logs instead (download
  progress prints there); `/health` only becomes available once both models have finished loading.
- **VRAM threshold**: only when **both** `QWEN_TTS_VOICE_DESIGN_ENABLED` and `QWEN_TTS_VOICE_CLONE_ENABLED`
  are `true` (the default) does the startup VRAM warning threshold bump to `QWEN_TTS_MIN_FREE_VRAM_GB + 6`
  (≈12GB by default), to account for both checkpoints loaded at once. With exactly one of the two enabled,
  the threshold stays at `QWEN_TTS_MIN_FREE_VRAM_GB` (single checkpoint). With both disabled, no model
  loads and the VRAM check is skipped entirely.
- **Persistent voices**: voices registered via `POST /v1/voices` persist in `QWEN_TTS_VOICES_DIR` (default `./voices`)
  across server restarts. This differs from job results, which are wiped on every startup.
- **Monitoring**: check `/health` response for `model_loaded` and `clone_model_loaded` flags.
  `vram_free_gb` shows available VRAM after both models are loaded.
- **Throughput**: trộn nhiều `voice_id` khác nhau trong cùng thời điểm làm giảm hiệu quả GPU (mỗi batch
  phải tách thành nhiều lần gọi model theo từng giọng); để throughput tốt nhất, chạy các job cùng một
  giọng tuần tự.

## Operational notes (Voice Design kill switch)

Khi `QWEN_TTS_VOICE_DESIGN_ENABLED=false` (ví dụ server production chỉ dùng
voice cloning): server chỉ tải **một checkpoint** (Base, cho voice cloning),
tiết kiệm ~4-5GB VRAM so với chạy song song hai model. `/health` báo
`model_loaded: false`, `clone_model_loaded: true`. Request `instruct`
(voice-design) sẽ fail từng item với `"voice design is disabled"` (`500`
với endpoint đồng bộ; item `failed` với Jobs API) — request `voice_id`
(clone) vẫn hoạt động bình thường. `python scripts/smoke_test.py` gọi
endpoint `/v1/tts/voice-design` và assert `model_loaded: true`, nên
**không áp dụng được** ở chế độ clone-only.

Nếu tắt cả hai (`QWEN_TTS_VOICE_DESIGN_ENABLED=false` và
`QWEN_TTS_VOICE_CLONE_ENABLED=false`), server khởi động mà không tải model
nào (chỉ log warning), và mọi request TTS sẽ fail.

## Manual verification

After starting the server, run `python scripts/smoke_test.py` to confirm the model loaded and a real WAV file comes back.

Optionally, use `python scripts/register_voices.py /path/to/voice_dir [base_url]` to bulk-register
a directory of `{name}.wav` + `{name}.txt` pairs as voices.
