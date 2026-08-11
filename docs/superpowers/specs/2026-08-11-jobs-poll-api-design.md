# Jobs + Poll API — Design

Date: 2026-08-11
Status: Approved

## Purpose

The client app generates large batches of TTS audio (a real production run:
578 shorts, each one text segment). With only the synchronous
`POST /v1/tts/voice-design` endpoint, the client must hold 578 concurrent
long-lived HTTP connections (each blocking up to 120s) or serialize them —
both fragile on a network that already drops ~20% of requests.

This feature adds an **asynchronous job API**: the client submits one job
containing many items, gets a `job_id` immediately, polls one endpoint for
per-item progress, and downloads each item's WAV individually as soon as
that item finishes. One job replaces hundreds of long-lived connections
with cheap short polls + downloads that are individually retryable.

## Decisions (made explicitly during design)

- **The synchronous endpoint stays.** `POST /v1/tts/voice-design` keeps
  working unchanged (convenient for quick tests / curl; no breaking change
  for existing clients). Jobs are additive.
- **1 job = many items** (list of text/language/instruct), max
  `MAX_ITEMS_PER_JOB` (default 1000) per job. No limit on the number of
  jobs — concurrent jobs interleave naturally through the shared queue.
- **Storage: job metadata in memory, audio on disk** under
  `RESULTS_DIR/{job_id}/{index}.wav`. No new dependencies (no SQLite).
  Jobs do not survive a server restart — the client resubmits.
- **Retention: results are deleted on server startup** (wipe `RESULTS_DIR`
  at boot). No TTL timer, no background cleanup task.
- **Download: per-item only** (`GET /v1/jobs/{job_id}/items/{index}/audio`).
  No ZIP endpoint. The client can fetch each item as soon as it's done and
  retry individual files on network errors.
- **Scheduling: job items go through the SAME `BatchWorker` queue** as
  synchronous requests. This reuses batching + audio self-check untouched,
  and means synchronous requests naturally interleave with running jobs.
  To keep a large job from monopolizing the queue, a job keeps at most
  `MAX_BATCH_SIZE` items in flight at once (semaphore).

## API

All existing endpoints are unchanged. New endpoints:

### `POST /v1/jobs` → `202 Accepted`

Request:
```json
{
  "items": [
    {"text": "Đoạn 1...", "language": "Auto", "instruct": "Giọng nữ trẻ..."},
    {"text": "Đoạn 2...", "language": "Auto", "instruct": "Giọng nữ trẻ..."}
  ]
}
```

- Each item is validated with the existing `TTSRequest` model (same rules:
  text non-empty ≤2000 chars, instruct non-empty, language whitelist).
  Any invalid item → the whole submit fails with `422` (nothing enqueued).
- Empty `items` list → `422`.
- More than `MAX_ITEMS_PER_JOB` items → `422`.

Response `202`:
```json
{"job_id": "j_a1b2c3d4", "status": "pending", "total_items": 2}
```

`job_id` format: `"j_"` + random hex (uuid4-derived), unguessable enough
for an internal unauthenticated server.

### `GET /v1/jobs/{job_id}` → `200`

```json
{
  "job_id": "j_a1b2c3d4",
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

- Job `status` lifecycle: `pending` → `running` → one of `completed`
  (all items done) | `completed_with_errors` (≥1 item failed) |
  `cancelled`.
- Item `status`: `pending` | `running` | `done` | `failed` (with `error`
  message) | `cancelled` (item never ran because the job was cancelled).
- `items` always contains one entry per item, in submission order.
- Unknown `job_id` → `404 {"detail": "job not found"}`.

### `GET /v1/jobs/{job_id}/items/{index}/audio` → `200 audio/wav`

- Returns the WAV bytes for one finished item (read from disk).
- Unknown job → `404`. Index out of range → `404`. Item not in `done`
  status (still pending/running, or failed/cancelled) → `409
  {"detail": "item not ready: <status>"}`.

### `DELETE /v1/jobs/{job_id}` → `200`

- Marks the job cancelled. Items currently in the GPU finish naturally
  (their results remain downloadable); items still `pending` become
  `cancelled` and never run.
- Idempotent: cancelling an already-finished or already-cancelled job
  returns `200` with the current state, changes nothing.
- Response body: same shape as `GET /v1/jobs/{job_id}`.
- Unknown `job_id` → `404`.

## Architecture

```
POST /v1/jobs
   │  validate all items (TTSRequest) → create Job(pending) → spawn asyncio task
   ▼
JobManager (app/jobs.py, new)
   ├─ jobs: dict[str, Job]                  (in-memory, lost on restart)
   ├─ per-job runner task:
   │     semaphore = asyncio.Semaphore(MAX_BATCH_SIZE)
   │     for each item (submission order):
   │        if job.cancelled: mark remaining pending items cancelled; stop
   │        async with semaphore:
   │           item.status = running
   │           wav = await batch_worker.submit(item.request)   ← SAME queue
   │           write RESULTS_DIR/{job_id}/{index}.wav
   │           item.status = done
   │        (exceptions from submit → item.status = failed + error message,
   │         job continues with the next item)
   │     job.status = completed | completed_with_errors | cancelled
   ▼
BatchWorker (unchanged) → TTSModelService (unchanged, incl. self-check)
```

- The runner launches items concurrently up to the semaphore limit
  (`MAX_BATCH_SIZE`, default 4), so a job's items can actually fill one
  GPU batch — but never more than one batch's worth — leaving room for
  synchronous requests to interleave.
- Per-item failures (e.g. audio self-check exhausted → per-item
  `RuntimeError` from the batcher) mark only that item `failed`; the job
  continues. This composes directly with the per-item failure isolation
  already built into `BatchWorker`/`TTSModelService`.
- Job runner tasks are tracked by `JobManager` and cancelled on server
  shutdown (lifespan), before `batch_worker.stop()` runs.

## Storage layout

```
RESULTS_DIR/                 (default ./results, env QWEN_TTS_RESULTS_DIR)
  j_a1b2c3d4/
    0.wav
    1.wav
    ...
```

- `RESULTS_DIR` is wiped and recreated at server startup (lifespan), per
  the retention decision.
- `results/` is added to `.gitignore`.

## New settings (`app/config.py`, env prefix `QWEN_TTS_`)

| Field | Type | Default |
|---|---|---|
| `max_items_per_job` | `int` | `1000` |
| `results_dir` | `str` | `"./results"` |

## Error handling

- Submit-time validation errors → `422` (FastAPI/pydantic, nothing enqueued).
- Item generation failure (any exception out of `batch_worker.submit`,
  including self-check exhaustion and queue timeouts) → that item is
  `failed` with the exception message; the job keeps going.
- Disk write failure for one item → that item `failed`, job keeps going.
- Server shutdown mid-job: runner tasks are cancelled in lifespan; since
  jobs don't survive restart anyway, no persistence of partial state is
  attempted. In-flight GPU work is handled by the existing
  `batch_worker.stop()` semantics.
- The audio download endpoint reads from disk only — it never touches the
  GPU queue and stays fast regardless of load (same principle as `/health`).

## Testing

- **Unit tests for `JobManager`** with a fake `batch_worker.submit`
  (no GPU): happy path (all items done, files written, statuses correct);
  per-item failure (one item fails, siblings continue, final status
  `completed_with_errors`); cancellation mid-job (pending items become
  `cancelled`, done items keep their files); semaphore bound (no more than
  `MAX_BATCH_SIZE` items in flight at once — assert via instrumented fake).
- **Endpoint tests** (existing `TestClient` harness from
  `tests/test_main.py`): submit → poll → download round-trip with a fake
  generate_fn; 422 on empty/oversized/invalid items; 404 unknown job;
  409 downloading an unfinished item; DELETE cancels and is idempotent.
- **Startup wipe test**: pre-create a stale `RESULTS_DIR` with files,
  start the app (TestClient lifespan), assert it was emptied.

## Out of scope

- Job persistence across restarts (client resubmits).
- ZIP/bulk download.
- Job priorities / preemption (single shared FIFO queue is the answer at
  this scale).
- Webhooks/server-push notifications (polling only).
- Authentication (unchanged: internal server).
- Word/sentence timestamps and fixed-voice features (separate plans, per
  the agreed priority order).
