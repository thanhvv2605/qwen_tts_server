# Qwen3-TTS VoiceDesign Server — API Reference

Base URL mặc định: `http://<host>:8000` (đổi qua `QWEN_TTS_HOST` / `QWEN_TTS_PORT`
nếu chạy bằng `python -m app.main`; nếu chạy `uvicorn app.main:app --host ... --port ...`
thì dùng đúng host/port đã truyền trên dòng lệnh).

Server không yêu cầu xác thực (API key/token) — chỉ dùng nội bộ/local.

---

## 1. `POST /v1/tts/voice-design`

Sinh audio từ text theo mô tả giọng nói mong muốn (voice design).

### Request

| Field      | Type   | Bắt buộc | Mặc định | Ghi chú                                                        |
|------------|--------|----------|----------|-----------------------------------------------------------------|
| `text`     | string | có       | —        | Nội dung cần đọc. Không được rỗng, tối đa 2000 ký tự.           |
| `language` | string | không    | `"Auto"` | Một trong: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian` |
| `instruct` | string | có       | —        | Mô tả giọng nói mong muốn (giới tính, tuổi, cảm xúc, tốc độ...). Không được rỗng. |

### curl mẫu

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Xin chào, đây là giọng nói được thiết kế riêng.",
    "language": "Auto",
    "instruct": "Giọng nữ trẻ, âm vui vẻ, tốc độ nói vừa phải."
  }' \
  -o output.wav
```

Lưu response trực tiếp thành file WAV bằng `-o output.wav`.

### Response `200 OK`

Trả về **binary WAV** (`Content-Type: audio/wav`), không phải JSON.

```
HTTP/1.1 200 OK
content-type: audio/wav
content-length: 153644

<binary WAV data>
```

Ví dụ header thật lấy từ server đang chạy:

```
HTTP/1.1 200 OK
date: Mon, 10 Aug 2026 15:22:05 GMT
server: uvicorn
content-length: 153644
content-type: audio/wav
```

### Response lỗi

#### `422 Unprocessable Entity` — input không hợp lệ

Validation của FastAPI/Pydantic tự động chạy trước khi tới batch worker.

**`text` rỗng:**
```bash
curl -X POST http://127.0.0.1:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{"text": "", "language": "English", "instruct": "calm voice"}'
```
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, text must not be empty",
      "input": "",
      "ctx": { "error": {} }
    }
  ]
}
```

**`language` không được hỗ trợ:**
```bash
curl -X POST http://127.0.0.1:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "language": "Klingon", "instruct": "calm voice"}'
```
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "language"],
      "msg": "Value error, language must be one of ['Auto', 'Chinese', 'English', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Portuguese', 'Russian', 'Spanish']",
      "input": "Klingon",
      "ctx": { "error": {} }
    }
  ]
}
```

**Thiếu field bắt buộc (`instruct`):**
```bash
curl -X POST http://127.0.0.1:8000/v1/tts/voice-design \
  -H "Content-Type: application/json" \
  -d '{"text": "hello", "language": "English"}'
```
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "instruct"],
      "msg": "Field required",
      "input": { "text": "hello", "language": "English" }
    }
  ]
}
```

#### `500 Internal Server Error` — lỗi khi sinh audio

Xảy ra khi model/batch worker gặp lỗi trong lúc generate (ví dụ lỗi nội bộ của model).

```json
{
  "detail": "<thông báo lỗi cụ thể>"
}
```

#### `504 Gateway Timeout` — hết thời gian chờ trong hàng đợi GPU

Request phải chờ quá `QWEN_TTS_REQUEST_TIMEOUT_S` (mặc định 120s) mà chưa được xử lý xong — thường do GPU đang quá tải.

```json
{
  "detail": "Timed out waiting for GPU queue"
}
```

---

## 2. `GET /health`

Kiểm tra trạng thái server. **Luôn trả lời nhanh**, không đi qua hàng đợi batch (không bị chặn dù server đang bận xử lý TTS).

### curl mẫu

```bash
curl http://127.0.0.1:8000/health
```

### Response `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true,
  "vram_free_gb": 18.2886962890625,
  "queue_depth": 0
}
```

| Field          | Type          | Ý nghĩa                                                             |
|----------------|---------------|----------------------------------------------------------------------|
| `status`       | string        | Luôn là `"ok"` nếu server còn phản hồi được.                        |
| `model_loaded` | boolean       | Model đã load xong lên GPU hay chưa (`false` trong lúc server đang khởi động/tải model). |
| `vram_free_gb` | number\|null  | VRAM còn trống (GB). `null` nếu model chưa load xong.                |
| `queue_depth`  | integer       | Số request đang chờ/đang xử lý trong batch worker (0 = rảnh).        |

---

## Ghi chú chung

- Tất cả cấu hình server (`QWEN_TTS_*`) xem ở `README.md` / `app/config.py`.
- Không có endpoint streaming ở phiên bản này — response WAV trả về đầy đủ sau khi generate xong.
- Nhiều request gửi gần nhau (trong `QWEN_TTS_BATCH_WINDOW_MS`, mặc định 150ms) sẽ được gộp lại xử lý cùng lúc trên GPU (tối đa `QWEN_TTS_MAX_BATCH_SIZE`, mặc định 4 request/batch) — không cần client tự làm gì, hoàn toàn tự động ở phía server.
