# Qwen3-TTS VoiceDesign Server — Design

Date: 2026-08-10
Status: Approved

## Purpose

Xây một REST API server nội bộ để sinh giọng nói (TTS) bằng model
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, cho phép mô tả giọng nói mong muốn
bằng ngôn ngữ tự nhiên (voice design) thay vì dùng giọng có sẵn.

## Constraints & context

- Máy chạy 1 GPU RTX 3090 24GB. Tại thời điểm thiết kế, VRAM có lúc bị
  tiến trình khác chiếm gần hết — server phải tự kiểm tra và cảnh báo,
  không tự ý kill tiến trình khác.
- Dùng cho 1 người/team nhỏ, không public ra internet → không cần auth.
- Ưu tiên setup nhanh gọn: bỏ qua FlashAttention 2 (build lâu, dễ lỗi),
  dùng attention mặc định của Transformers.
- Chỉ hỗ trợ model VoiceDesign (không CustomVoice/Base/Clone) ở v1.

## Architecture

```
Client A ──┐
Client B ──┼─ POST /v1/tts/voice-design {text, language, instruct}
Client C ──┘
        │
        ▼
FastAPI endpoint: tạo asyncio.Future, đẩy (request, future) vào asyncio.Queue
        │
        ▼
Background batching worker (1 task duy nhất, sống suốt vòng đời server):
  1. Lấy request đầu tiên từ queue (chờ vô thời hạn nếu queue rỗng)
  2. Gom thêm request đến trong tối đa BATCH_WINDOW_MS (150ms) HOẶC đến khi
     đủ MAX_BATCH_SIZE (4) request
  3. Gọi model.generate_voice_design(text=[...], language=[...], instruct=[...])
     trong threadpool executor (không block event loop)
  4. Phân phối kết quả (hoặc lỗi) về đúng Future của từng client đang chờ
        │
        ▼
Qwen3TTSModel (load 1 lần lúc startup, singleton, giữ trong VRAM)
        │
        ▼
Mỗi client nhận lại WAV bytes riêng của mình → StreamingResponse audio/wav
```

Chỉ có **1 worker duy nhất** tiêu thụ queue → GPU không bao giờ nhận 2 batch
chồng nhau, tránh OOM VRAM.

Trade-off chấp nhận: câu ngắn trong cùng batch phải chờ câu dài nhất sinh
xong (do padding + generation length bị chi phối bởi phần tử dài nhất
trong batch). Chấp nhận được vì window ngắn (150ms) và batch nhỏ (≤4).

Nếu cả batch lỗi (model raise exception giữa chừng), toàn bộ request trong
batch đó nhận lỗi 500 — không retry tách lẻ từng item ở v1.

## Components

| File | Trách nhiệm |
|---|---|
| `app/config.py` | Settings qua env vars: `MODEL_ID` (mặc định `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`), `DEVICE` (mặc định `cuda:0`), `HOST` (`0.0.0.0`), `PORT` (`8000`), `MIN_FREE_VRAM_GB` (`6`), `BATCH_WINDOW_MS` (`150`), `MAX_BATCH_SIZE` (`4`), `MAX_NEW_TOKENS` (`2048`), `REQUEST_TIMEOUT_S` (`120`) |
| `app/schemas.py` | Pydantic `TTSRequest` (text: str, language: str = "Auto", instruct: str) với validation; error response model |
| `app/model.py` | `TTSModelService`: kiểm tra VRAM + load singleton `Qwen3TTSModel` lúc startup; hàm `generate_batch(items: list[TTSRequest]) -> list[bytes]` chạy blocking call trong threadpool |
| `app/batcher.py` | `BatchWorker`: `asyncio.Queue` + 1 background task duy nhất, gom request theo window/max-size rồi gọi `TTSModelService.generate_batch`, phân phối kết quả về từng `asyncio.Future` |
| `app/main.py` | FastAPI app + lifespan (load model, start/stop batch worker) + endpoints |
| `requirements.txt` | Dependencies |
| `README.md` | Hướng dẫn cài đặt & chạy |
| `scripts/smoke_test.py` | Test thủ công sau khi server chạy thật với model thật |
| `tests/` | Unit test cho batching logic và validation (không cần GPU) |

## API

### `POST /v1/tts/voice-design`

Request JSON:
```json
{
  "text": "Xin chào, hôm nay trời đẹp quá.",
  "language": "Auto",
  "instruct": "Giọng nữ trẻ, vui vẻ, tốc độ nói nhanh."
}
```

Response: `audio/wav` binary (200) hoặc:
- `422` — input không hợp lệ (text/instruct rỗng, language không hỗ trợ)
- `504` — hết thời gian chờ trong hàng đợi (> `REQUEST_TIMEOUT_S`)
- `500` — lỗi khi sinh audio, body `{"error": "..."}`

### `GET /health`

Response `200`:
```json
{
  "status": "ok",
  "model_loaded": true,
  "vram_free_gb": 18.2,
  "queue_depth": 0
}
```
Endpoint này luôn trả nhanh, không đi qua batch queue.

## Data flow

Endpoint nhận request → validate Pydantic → tạo `asyncio.Future` → đẩy
`(request, future)` vào Queue → `await future` (timeout `REQUEST_TIMEOUT_S`)
→ BatchWorker gom & gọi model → trả WAV bytes qua
`StreamingResponse(media_type="audio/wav")`.

## Error handling

- **Validate input**: `text` không rỗng, giới hạn độ dài (2000 ký tự);
  `instruct` bắt buộc không rỗng (voice design cần mô tả giọng để có ý
  nghĩa); `language` nằm trong danh sách 10 ngôn ngữ hỗ trợ hoặc `"Auto"`.
  Sai → `422`.
- **Startup VRAM check**: dùng `torch.cuda.mem_get_info()` trước khi load
  model. Nếu free < `MIN_FREE_VRAM_GB`, log WARNING liệt kê tiến trình
  đang chiếm VRAM (qua `nvidia-smi`), vẫn thử load. Nếu
  `torch.cuda.OutOfMemoryError` xảy ra khi load → log rõ nguyên nhân +
  gợi ý giải phóng VRAM, server thoát với exit code ≠ 0 (không chạy nửa
  vời, không âm thầm fallback sang CPU).
- **Batch lỗi**: exception khi `generate_batch` → set exception cho toàn
  bộ Future đang chờ trong batch đó; log traceback kèm text (rút gọn) để
  debug. Client nhận `500 {"error": ...}`.
- **Timeout hàng đợi**: quá `REQUEST_TIMEOUT_S` chưa được xử lý → hủy chờ,
  trả `504`.

## Testing

- **Unit test cho `BatchWorker`** (không cần GPU/model thật): dependency-
  inject một hàm generate giả lập trả về ngay; dùng `pytest-asyncio` mô
  phỏng nhiều request đến gần nhau để verify đúng logic gom theo
  window/max-size, và verify lỗi trong batch lan đúng tới toàn bộ Future
  liên quan.
- **Unit test cho `TTSRequest` validation**: text rỗng, instruct rỗng,
  language không hợp lệ → đúng lỗi 422.
- **Smoke test thủ công** (`scripts/smoke_test.py`) chạy sau khi server
  thật (có GPU + model đã tải) đang chạy: gọi `/health` và
  `/v1/tts/voice-design`, kiểm tra response là WAV hợp lệ bằng
  `soundfile.read`.

## Setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts fastapi "uvicorn[standard]" soundfile pydantic-settings pytest pytest-asyncio httpx
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Model tự động tải từ HuggingFace (~4GB) lần chạy đầu tiên, hoặc tải trước
bằng `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
--local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` và trỏ `MODEL_ID`
tới đường dẫn local đó.

## Out of scope (v1)

- Streaming audio response (dù model hỗ trợ streaming generation).
- CustomVoice / Base (voice clone) models.
- Authentication / rate limiting.
- Lưu lịch sử audio đã tạo.
- Retry tách lẻ từng item khi cả batch lỗi.
