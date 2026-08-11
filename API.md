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

Xảy ra khi model/batch worker gặp lỗi trong lúc generate (ví dụ lỗi nội bộ
của model), hoặc khi **audio self-check** thất bại: server tự phát hiện
audio sinh ra ngắn bất thường so với độ dài text (audio bị cụt), tự sinh
lại tối đa `QWEN_TTS_AUDIO_SELF_CHECK_MAX_RETRIES` (mặc định 2) lần, và
chỉ trả `500` nếu vẫn không đạt. Client nên coi lỗi này là **retryable**
(gửi lại request).

```json
{
  "detail": "<thông báo lỗi cụ thể>"
}
```

Ví dụ message khi self-check thất bại:
```json
{
  "detail": "audio self-check failed after 2 retries: got 0.33s for 35 words (expected >= 7.78s)"
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

## 2. Jobs API (xử lý bất đồng bộ theo lô)

Dành cho lô lớn (ví dụ hàng trăm đoạn text): gửi 1 job, poll tiến độ, tải
từng file khi xong. Job **không** sống sót khi server restart (client gửi
lại), và toàn bộ kết quả cũ bị xóa mỗi lần server khởi động.

### `POST /v1/jobs` — tạo job

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Đoạn 1...", "language": "Auto", "instruct": "Giọng nữ trẻ..."},
      {"text": "Đoạn 2...", "language": "Auto", "instruct": "Giọng nữ trẻ..."}
    ]
  }'
```

Response `202`:
```json
{"job_id": "j_a1b2c3d4e5f6", "status": "pending", "total_items": 2}
```

- Mỗi item validate đúng như endpoint đồng bộ (text ≤2000 ký tự, instruct
  bắt buộc, language trong danh sách). Bất kỳ item nào sai → `422`, không
  item nào được nhận.
- Tối đa `QWEN_TTS_MAX_ITEMS_PER_JOB` (mặc định 1000) items/job → quá → `422`.

### `GET /v1/jobs/{job_id}` — poll tiến độ

```bash
curl http://127.0.0.1:8000/v1/jobs/j_a1b2c3d4e5f6
```

Response `200`:
```json
{
  "job_id": "j_a1b2c3d4e5f6",
  "status": "running",
  "total_items": 578,
  "done": 213,
  "failed": 1,
  "items": [
    {"index": 0, "status": "done"},
    {"index": 1, "status": "failed", "error": "audio self-check failed after 2 retries: ..."},
    {"index": 2, "status": "running"},
    {"index": 3, "status": "pending"}
  ]
}
```

- Job `status`: `pending` → `running` → `completed` |
  `completed_with_errors` | `cancelled`.
- Item `status`: `pending` | `running` | `done` | `failed` | `cancelled`.
- Job không tồn tại → `404`.
- `items` luôn chứa đủ toàn bộ N items (ví dụ trên đã rút gọn) — cân nhắc
  băng thông khi poll job lớn.

### `GET /v1/jobs/{job_id}/items/{index}/audio` — tải audio 1 item

```bash
curl http://127.0.0.1:8000/v1/jobs/j_a1b2c3d4e5f6/items/0/audio -o item0.wav
```

- Item `done` → `200` binary WAV. Tải được ngay khi item xong, không cần
  chờ cả job.
- Job không tồn tại hoặc index ngoài phạm vi → `404`.
- Item chưa xong (pending/running) hoặc failed/cancelled → `409`
  `{"detail": "item not ready: <status>"}`.
- Trường hợp hiếm `404 {"detail": "result file missing"}`: item báo `done`
  nhưng file WAV trên đĩa không còn (ví dụ bị xóa thủ công ngoài server) —
  phân biệt với `404` "job không tồn tại"/"index ngoài phạm vi" ở trên qua
  nội dung `detail`.

### `DELETE /v1/jobs/{job_id}` — hủy job

```bash
curl -X DELETE http://127.0.0.1:8000/v1/jobs/j_a1b2c3d4e5f6
```

- Item đang chạy trên GPU chạy nốt (kết quả vẫn tải được); item còn
  `pending` chuyển thành `cancelled`.
- Idempotent: hủy job đã xong/đã hủy → `200`, không đổi gì.
- Job không tồn tại → `404`.
- Response: cùng shape với `GET /v1/jobs/{job_id}`.
- **Response của `DELETE` là snapshot ngay tại thời điểm gọi, TRƯỚC khi
  cancel có hiệu lực** — item đang `running` vẫn hiện `running` trong
  response này dù sắp bị đánh dấu `cancelled`. Phải poll tiếp
  `GET /v1/jobs/{job_id}` đến khi `status` chuyển hẳn sang `cancelled` để
  biết trạng thái cuối cùng.

---

## 3. `GET /health`

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
