# Jobs + Poll API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an asynchronous job API — submit one job with up to 1000 TTS
items, poll one endpoint for per-item progress, download each item's WAV
individually as soon as it's done — so bulk clients (e.g. 578 shorts) stop
holding hundreds of long-lived HTTP connections.

**Architecture:** A new `JobManager` (`app/jobs.py`) keeps job metadata in
memory and writes finished WAVs to `RESULTS_DIR/{job_id}/{index}.wav`. Each
job runs as one `asyncio.Task` that feeds items into the EXISTING
`BatchWorker` queue (`batch_worker.submit`), capped at `MAX_BATCH_SIZE`
in-flight items per job via a semaphore so synchronous requests interleave
naturally. Per-item failures (self-check exhaustion, timeouts) mark only
that item failed; the job continues. `RESULTS_DIR` is wiped at server
startup (jobs don't survive restarts). The synchronous endpoint is
unchanged.

**Tech Stack:** Same as the existing project (Python 3.12, FastAPI, pytest +
pytest-asyncio, stdlib `uuid`/`shutil`/`pathlib`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-11-jobs-poll-api-design.md`

## Global Constraints

- `MAX_ITEMS_PER_JOB` default `1000` (env `QWEN_TTS_MAX_ITEMS_PER_JOB`); submit with more items → `422`; empty items list → `422`
- `RESULTS_DIR` default `"./results"` (env `QWEN_TTS_RESULTS_DIR`); wiped and recreated at server startup; added to `.gitignore`
- Each job item is validated with the EXISTING `TTSRequest` model — same rules as the sync endpoint; any invalid item fails the whole submit with `422`, nothing enqueued
- Job items go through the SAME `BatchWorker` queue as sync requests; a job keeps at most `settings.max_batch_size` items in flight at once
- Per-item generation failure (any exception from `batch_worker.submit`, including per-item self-check failures and the `request_timeout_s` timeout) marks only that item `failed`; the job continues
- Job status lifecycle: `pending` → `running` → `completed` | `completed_with_errors` | `cancelled`. Item status: `pending` | `running` | `done` | `failed` | `cancelled`
- `job_id` format: `"j_"` + 12 hex chars from uuid4
- Endpoints and status codes (exact): `POST /v1/jobs` → 202; `GET /v1/jobs/{job_id}` → 200/404; `GET /v1/jobs/{job_id}/items/{index}/audio` → 200 (`audio/wav`)/404 (unknown job or index out of range)/409 (item not `done`, detail `"item not ready: <status>"`); `DELETE /v1/jobs/{job_id}` → 200/404, idempotent
- Cancel: in-flight items finish naturally (results stay downloadable); `pending` items become `cancelled`
- The audio download endpoint reads from disk only — never touches the GPU queue
- No changes to `app/batcher.py` or `app/model.py`
- The existing `/v1/tts/voice-design` and `/health` endpoints are unchanged

---

## Task 1: Settings and job submit schema

**Files:**
- Modify: `app/config.py`
- Modify: `app/schemas.py`
- Modify: `.gitignore`
- Modify: `tests/test_config.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**
- Produces: `settings.max_items_per_job: int = 1000`, `settings.results_dir: str = "./results"`; `JobSubmitRequest(BaseModel)` in `app/schemas.py` with `items: list[TTSRequest]` (min length 1 enforced by pydantic). The per-job max is NOT enforced in the schema (it needs `settings`) — Task 3's endpoint enforces it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, extend `test_default_settings` with:
```python
    assert settings.max_items_per_job == 1000
    assert settings.results_dir == "./results"
```
and add:
```python
def test_env_override_job_settings(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_MAX_ITEMS_PER_JOB", "50")
    monkeypatch.setenv("QWEN_TTS_RESULTS_DIR", "/tmp/tts-results")
    settings = Settings(_env_file=None)
    assert settings.max_items_per_job == 50
    assert settings.results_dir == "/tmp/tts-results"
```

In `tests/test_schemas.py`, add (`pytest` and `ValidationError` are already imported there):
```python
from app.schemas import JobSubmitRequest


def test_job_submit_request_valid():
    req = JobSubmitRequest(
        items=[
            {"text": "hello", "language": "English", "instruct": "calm voice"},
            {"text": "world", "instruct": "excited voice"},
        ]
    )
    assert len(req.items) == 2
    assert req.items[0].text == "hello"
    assert req.items[1].language == "Auto"


def test_job_submit_request_rejects_empty_items():
    with pytest.raises(ValidationError):
        JobSubmitRequest(items=[])


def test_job_submit_request_rejects_invalid_item():
    with pytest.raises(ValidationError):
        JobSubmitRequest(items=[{"text": "", "language": "English", "instruct": "calm"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_schemas.py -v`
Expected: FAIL — `AttributeError` on the new settings fields; `ImportError: cannot import name 'JobSubmitRequest'`

- [ ] **Step 3: Write minimal implementation**

`app/config.py` — add after `audio_self_check_enabled`:
```python
    audio_self_check_enabled: bool = True
    max_items_per_job: int = 1000
    results_dir: str = "./results"
```

`app/schemas.py` — change the import line and add the new model at the end of the file:
```python
from pydantic import BaseModel, Field, field_validator
```
```python
class JobSubmitRequest(BaseModel):
    items: list[TTSRequest] = Field(min_length=1)
```

`.gitignore` — append:
```text
results/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py tests/test_schemas.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/schemas.py .gitignore tests/test_config.py tests/test_schemas.py
git commit -m "feat: add job settings and JobSubmitRequest schema"
```

---

## Task 2: JobManager

**Files:**
- Create: `app/jobs.py`
- Create: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `TTSRequest` from `app.schemas`; `Settings` from `app.config` (reads `results_dir` and `max_batch_size` dynamically so tests can override them on a `Settings` instance).
- Produces (Task 3 depends on all of these):
  - `ItemStatus` / `JobStatus` (str Enums with the exact values from Global Constraints)
  - `JobItem` (dataclass: `request: TTSRequest`, `status: ItemStatus`, `error: str | None`)
  - `Job` (dataclass: `job_id: str`, `items: list[JobItem]`, `status: JobStatus`, `cancel_requested: bool`, `task: asyncio.Task | None`)
  - `SubmitFn = Callable[[TTSRequest], Awaitable[bytes]]`
  - `JobManager(submit_fn: SubmitFn, settings: Settings)` with:
    - `wipe_results_dir() -> None` — rmtree + recreate `settings.results_dir`
    - `create_job(requests: list[TTSRequest]) -> Job` — creates the job dir, spawns the runner task, returns immediately with status `pending`
    - `get_job(job_id: str) -> Job | None`
    - `audio_path(job_id: str, index: int) -> Path`
    - `cancel_job(job_id: str) -> Job | None` — sets `cancel_requested` if still pending/running; idempotent; returns the job (or `None` if unknown)
    - `async shutdown() -> None` — cancels and awaits all unfinished runner tasks
  - `job_to_dict(job: Job) -> dict` — the poll-response shape (`job_id`, `status`, `total_items`, `done`, `failed`, `items` with per-item `index`/`status` and `error` only when set)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_jobs.py`:
```python
import asyncio

import pytest

from app.config import Settings
from app.jobs import ItemStatus, JobManager, JobStatus, job_to_dict
from app.schemas import TTSRequest


def _req(text: str) -> TTSRequest:
    return TTSRequest(text=text, language="English", instruct="calm voice")


def _settings(tmp_path, max_batch_size: int = 4) -> Settings:
    return Settings(_env_file=None, results_dir=str(tmp_path / "results"), max_batch_size=max_batch_size)


async def test_happy_path_all_items_done(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return f"wav-{request.text}".encode()

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a"), _req("b"), _req("c")])
    assert job.status is JobStatus.PENDING

    await job.task

    assert job.status is JobStatus.COMPLETED
    assert all(item.status is ItemStatus.DONE for item in job.items)
    for i, text in enumerate(["a", "b", "c"]):
        assert manager.audio_path(job.job_id, i).read_bytes() == f"wav-{text}".encode()


async def test_per_item_failure_keeps_job_going(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        if request.text == "bad":
            raise RuntimeError("audio self-check failed")
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("good1"), _req("bad"), _req("good2")])
    await job.task

    assert job.status is JobStatus.COMPLETED_WITH_ERRORS
    assert job.items[0].status is ItemStatus.DONE
    assert job.items[1].status is ItemStatus.FAILED
    assert "audio self-check failed" in job.items[1].error
    assert job.items[2].status is ItemStatus.DONE
    assert manager.audio_path(job.job_id, 0).exists()
    assert not manager.audio_path(job.job_id, 1).exists()


async def test_cancel_skips_pending_items_but_keeps_done_ones(tmp_path):
    release = asyncio.Event()

    async def slow_submit(request: TTSRequest) -> bytes:
        if request.text != "first":
            await release.wait()
        return b"ok"

    # max_batch_size=1 so items run strictly one at a time.
    manager = JobManager(submit_fn=slow_submit, settings=_settings(tmp_path, max_batch_size=1))
    manager.wipe_results_dir()
    job = manager.create_job([_req("first"), _req("second"), _req("third")])

    while job.items[0].status is not ItemStatus.DONE:
        await asyncio.sleep(0.01)

    manager.cancel_job(job.job_id)
    release.set()
    await job.task

    assert job.status is JobStatus.CANCELLED
    assert job.items[0].status is ItemStatus.DONE
    assert manager.audio_path(job.job_id, 0).exists()
    # "second" may have been in flight when cancel landed (done) or not
    # (cancelled); "third" must never have run.
    assert job.items[2].status is ItemStatus.CANCELLED


async def test_in_flight_capped_at_max_batch_size(tmp_path):
    in_flight = 0
    max_seen = 0
    release = asyncio.Event()

    async def tracking_submit(request: TTSRequest) -> bytes:
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await release.wait()
        in_flight -= 1
        return b"ok"

    manager = JobManager(submit_fn=tracking_submit, settings=_settings(tmp_path, max_batch_size=2))
    manager.wipe_results_dir()
    job = manager.create_job([_req(str(i)) for i in range(6)])

    await asyncio.sleep(0.05)
    release.set()
    await job.task

    assert job.status is JobStatus.COMPLETED
    assert max_seen == 2


async def test_wipe_results_dir_removes_stale_content(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    settings = _settings(tmp_path)
    manager = JobManager(submit_fn=fake_submit, settings=settings)
    stale = tmp_path / "results" / "old_job"
    stale.mkdir(parents=True)
    (stale / "0.wav").write_bytes(b"stale")

    manager.wipe_results_dir()

    results_root = tmp_path / "results"
    assert results_root.exists()
    assert list(results_root.iterdir()) == []


async def test_shutdown_cancels_running_jobs(tmp_path):
    started = asyncio.Event()

    async def hanging_submit(request: TTSRequest) -> bytes:
        started.set()
        await asyncio.Event().wait()  # never returns
        return b"unreachable"

    manager = JobManager(submit_fn=hanging_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a")])
    await started.wait()

    await manager.shutdown()

    assert job.task.done()
    assert job.status is JobStatus.CANCELLED
    assert job.items[0].status is ItemStatus.CANCELLED


async def test_job_to_dict_shape(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        if request.text == "bad":
            raise RuntimeError("boom")
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("good"), _req("bad")])
    await job.task

    d = job_to_dict(job)
    assert d["job_id"] == job.job_id
    assert d["status"] == "completed_with_errors"
    assert d["total_items"] == 2
    assert d["done"] == 1
    assert d["failed"] == 1
    assert d["items"][0] == {"index": 0, "status": "done"}
    assert d["items"][1]["index"] == 1
    assert d["items"][1]["status"] == "failed"
    assert "boom" in d["items"][1]["error"]


async def test_get_and_cancel_unknown_job(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    assert manager.get_job("j_nope") is None
    assert manager.cancel_job("j_nope") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.jobs'`

- [ ] **Step 3: Write minimal implementation**

Create `app/jobs.py`:
```python
import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from app.config import Settings
from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

SubmitFn = Callable[[TTSRequest], Awaitable[bytes]]


class ItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    CANCELLED = "cancelled"


@dataclass
class JobItem:
    request: TTSRequest
    status: ItemStatus = ItemStatus.PENDING
    error: str | None = None


@dataclass
class Job:
    job_id: str
    items: list[JobItem]
    status: JobStatus = JobStatus.PENDING
    cancel_requested: bool = False
    task: "asyncio.Task | None" = None


def job_to_dict(job: Job) -> dict:
    items = []
    for index, item in enumerate(job.items):
        entry: dict = {"index": index, "status": item.status.value}
        if item.error is not None:
            entry["error"] = item.error
        items.append(entry)
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "total_items": len(job.items),
        "done": sum(1 for i in job.items if i.status is ItemStatus.DONE),
        "failed": sum(1 for i in job.items if i.status is ItemStatus.FAILED),
        "items": items,
    }


class JobManager:
    def __init__(self, submit_fn: SubmitFn, settings: Settings) -> None:
        self._submit_fn = submit_fn
        self._settings = settings
        self._jobs: dict[str, Job] = {}

    def _results_root(self) -> Path:
        return Path(self._settings.results_dir)

    def wipe_results_dir(self) -> None:
        shutil.rmtree(self._results_root(), ignore_errors=True)
        self._results_root().mkdir(parents=True, exist_ok=True)

    def audio_path(self, job_id: str, index: int) -> Path:
        return self._results_root() / job_id / f"{index}.wav"

    def create_job(self, requests: list[TTSRequest]) -> Job:
        job_id = f"j_{uuid.uuid4().hex[:12]}"
        job = Job(job_id=job_id, items=[JobItem(request=r) for r in requests])
        self._jobs[job_id] = job
        (self._results_root() / job_id).mkdir(parents=True, exist_ok=True)
        job.task = asyncio.create_task(self._run_job(job))
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.cancel_requested = True
        return job

    async def shutdown(self) -> None:
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.task.cancel()
        for job in self._jobs.values():
            if job.task is not None:
                try:
                    await job.task
                except asyncio.CancelledError:
                    pass

    async def _run_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        semaphore = asyncio.Semaphore(self._settings.max_batch_size)

        async def run_item(index: int, item: JobItem) -> None:
            async with semaphore:
                if job.cancel_requested:
                    item.status = ItemStatus.CANCELLED
                    return
                item.status = ItemStatus.RUNNING
                try:
                    wav_bytes = await self._submit_fn(item.request)
                    path = self.audio_path(job.job_id, index)
                    await asyncio.get_running_loop().run_in_executor(
                        None, path.write_bytes, wav_bytes
                    )
                except asyncio.CancelledError:
                    item.status = ItemStatus.CANCELLED
                    raise
                except Exception as exc:  # noqa: BLE001 - per-item failure, job continues
                    logger.warning("job %s item %d failed: %s", job.job_id, index, exc)
                    item.status = ItemStatus.FAILED
                    item.error = str(exc)
                else:
                    item.status = ItemStatus.DONE

        try:
            await asyncio.gather(*(run_item(i, item) for i, item in enumerate(job.items)))
        except asyncio.CancelledError:
            for item in job.items:
                if item.status in (ItemStatus.PENDING, ItemStatus.RUNNING):
                    item.status = ItemStatus.CANCELLED
            job.status = JobStatus.CANCELLED
            raise
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
        elif any(item.status is ItemStatus.FAILED for item in job.items):
            job.status = JobStatus.COMPLETED_WITH_ERRORS
        else:
            job.status = JobStatus.COMPLETED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_jobs.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Run these concurrency-flavored tests repeatedly to check for flakiness**

Run: `for i in $(seq 1 10); do pytest tests/test_jobs.py -q || break; done`
Expected: 10 consecutive clean runs.

- [ ] **Step 6: Commit**

```bash
git add app/jobs.py tests/test_jobs.py
git commit -m "feat: add JobManager for async multi-item TTS jobs"
```

---

## Task 3: Endpoints, lifespan wiring, and shared test fixture

**Files:**
- Modify: `app/main.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_main.py`
- Create: `tests/test_jobs_api.py`

**Interfaces:**
- Consumes: `JobManager`, `ItemStatus`, `job_to_dict` from `app.jobs` (Task 2); `JobSubmitRequest` from `app.schemas` (Task 1); `settings.max_items_per_job` / `settings.results_dir` (Task 1); existing `batch_worker`, `settings`, `model_service`.
- Produces: module-level `job_manager: JobManager` in `app.main` (accessible as `app.main.job_manager` for tests); the four job endpoints; lifespan wiping `RESULTS_DIR` at startup and calling `job_manager.shutdown()` before `batch_worker.stop()` at shutdown.

**Why the fixture must move to `tests/conftest.py`:** `tests/test_main.py`'s
module-scoped `client` fixture exists because `batch_worker`'s internal
`asyncio.Queue` binds permanently to the first event loop that touches it,
and each `TestClient(...)` context entry creates a fresh loop. A SECOND test
module creating its own `TestClient` would hit the same
"Queue is bound to a different event loop" crash. Since this task adds a
second module of endpoint tests (`tests/test_jobs_api.py`), the fixture
must become **session-scoped and shared** via `tests/conftest.py` — one
`TestClient` (one loop, one lifespan cycle) for the entire test session,
matching how the real server runs.

- [ ] **Step 1: Move the client fixture to a session-scoped conftest**

Create `tests/conftest.py`:
```python
import io

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from app import main as main_module


def fake_wav_bytes() -> bytes:
    wav = np.zeros(2400, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    # Pre-seed a stale file so a test can verify the startup wipe.
    d = tmp_path_factory.mktemp("results")
    (d / "stale.txt").write_text("old")
    return d


@pytest.fixture(scope="session")
def client(results_dir):
    # Session-scoped (one TestClient, one event loop, one lifespan cycle for
    # the whole test session): batch_worker's internal asyncio.Queue binds
    # permanently to the first event loop that uses it, so a second
    # TestClient anywhere in the suite would crash with "Queue ... is bound
    # to a different event loop". One shared client also matches how the
    # real server runs (lifespan started once per process).
    mp = pytest.MonkeyPatch()
    mp.setattr(main_module.settings, "results_dir", str(results_dir))
    mp.setattr(main_module.model_service, "load", lambda: None)
    mp.setattr(main_module.model_service, "is_loaded", lambda: True)
    mp.setattr("app.model.check_vram", lambda device, min_free_gb: 20.0)

    async def fake_generate_fn(requests):
        return [fake_wav_bytes() for _ in requests]

    mp.setattr(main_module.batch_worker, "_generate_fn", fake_generate_fn)

    try:
        with TestClient(main_module.app) as test_client:
            yield test_client
    finally:
        mp.undo()
```

In `tests/test_main.py`, DELETE the module-scoped `client` fixture (lines
19-43) and the now-unused imports it leaves behind: remove
`import numpy as np`, `import pytest`, and
`from starlette.testclient import TestClient`; keep `import asyncio`,
`import io`, `import soundfile as sf`, and
`from app import main as main_module` (the 500/504 tests still use
`main_module` and the `monkeypatch` fixture, and the WAV test still uses
`io`/`sf`). Replace the local `_fake_wav_bytes` helper with an import:
```python
from tests.conftest import fake_wav_bytes as _fake_wav_bytes
```
All five test function bodies stay unchanged.

- [ ] **Step 2: Verify the move broke nothing**

Run: `pytest tests/test_main.py -v`
Expected: PASS (5 passed) — same tests, fixture now coming from conftest.

- [ ] **Step 3: Write the failing endpoint tests**

Create `tests/test_jobs_api.py`:
```python
import io
import time

import soundfile as sf

from app import main as main_module


def _item(text: str) -> dict:
    return {"text": text, "language": "English", "instruct": "calm voice"}


def _wait_until_finished(client, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] not in ("pending", "running"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s: {body}")


def test_startup_wiped_stale_results(client, results_dir):
    assert not (results_dir / "stale.txt").exists()


def test_submit_poll_download_roundtrip(client):
    resp = client.post("/v1/jobs", json={"items": [_item("hello"), _item("world")]})
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"].startswith("j_")
    assert body["total_items"] == 2

    final = _wait_until_finished(client, body["job_id"])
    assert final["status"] == "completed"
    assert final["done"] == 2
    assert final["failed"] == 0
    assert final["items"][0]["status"] == "done"
    assert final["items"][1]["status"] == "done"

    audio = client.get(f"/v1/jobs/{body['job_id']}/items/0/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(audio.content))
    assert sr == 24000


def test_submit_rejects_empty_items(client):
    resp = client.post("/v1/jobs", json={"items": []})
    assert resp.status_code == 422


def test_submit_rejects_invalid_item(client):
    resp = client.post(
        "/v1/jobs",
        json={"items": [_item("ok"), {"text": "", "language": "English", "instruct": "x"}]},
    )
    assert resp.status_code == 422


def test_submit_rejects_too_many_items(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "max_items_per_job", 2)
    resp = client.post("/v1/jobs", json={"items": [_item("a"), _item("b"), _item("c")]})
    assert resp.status_code == 422
    assert "at most 2" in resp.json()["detail"]


def test_get_unknown_job_returns_404(client):
    resp = client.get("/v1/jobs/j_doesnotexist")
    assert resp.status_code == 404


def test_download_unfinished_item_returns_409(client, monkeypatch):
    import asyncio

    async def hanging_generate_fn(requests):
        await asyncio.sleep(30)
        return [b"" for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", hanging_generate_fn)

    resp = client.post("/v1/jobs", json={"items": [_item("slow")]})
    job_id = resp.json()["job_id"]

    audio = client.get(f"/v1/jobs/{job_id}/items/0/audio")
    assert audio.status_code == 409
    assert "item not ready" in audio.json()["detail"]

    out_of_range = client.get(f"/v1/jobs/{job_id}/items/5/audio")
    assert out_of_range.status_code == 404

    client.delete(f"/v1/jobs/{job_id}")


def test_cancel_job_and_idempotency(client, monkeypatch):
    import asyncio

    async def slow_generate_fn(requests):
        await asyncio.sleep(0.2)
        from tests.conftest import fake_wav_bytes

        return [fake_wav_bytes() for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", slow_generate_fn)

    resp = client.post("/v1/jobs", json={"items": [_item(str(i)) for i in range(10)]})
    job_id = resp.json()["job_id"]

    cancel = client.delete(f"/v1/jobs/{job_id}")
    assert cancel.status_code == 200

    final = _wait_until_finished(client, job_id)
    assert final["status"] == "cancelled"
    assert any(item["status"] == "cancelled" for item in final["items"])

    again = client.delete(f"/v1/jobs/{job_id}")
    assert again.status_code == 200
    assert again.json()["status"] == "cancelled"


def test_cancel_unknown_job_returns_404(client):
    resp = client.delete("/v1/jobs/j_doesnotexist")
    assert resp.status_code == 404
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_jobs_api.py -v`
Expected: FAIL — 404s from FastAPI for the not-yet-existing `/v1/jobs` routes (assertion errors on status codes).

- [ ] **Step 5: Write minimal implementation**

In `app/main.py`:

Change the imports block to:
```python
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from app import model as model_module
from app.batcher import BatchWorker
from app.config import settings
from app.jobs import ItemStatus, JobManager, job_to_dict
from app.model import TTSModelService
from app.schemas import JobSubmitRequest, TTSRequest
```

After the `batch_worker = BatchWorker(...)` block, add:
```python
async def _job_submit(request: TTSRequest) -> bytes:
    return await asyncio.wait_for(
        batch_worker.submit(request), timeout=settings.request_timeout_s
    )


job_manager = JobManager(submit_fn=_job_submit, settings=settings)
```

Replace the lifespan function with:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.wipe_results_dir()
    model_service.load()
    batch_worker.start()
    yield
    await job_manager.shutdown()
    await batch_worker.stop()
```

Add the four endpoints after the existing `health()` endpoint (and before the `if __name__ == "__main__":` block):
```python
@app.post("/v1/jobs", status_code=202)
async def submit_job(request: JobSubmitRequest) -> dict:
    if len(request.items) > settings.max_items_per_job:
        raise HTTPException(
            status_code=422,
            detail=f"a job may contain at most {settings.max_items_per_job} items",
        )
    job = job_manager.create_job(request.items)
    return {"job_id": job.job_id, "status": job.status.value, "total_items": len(job.items)}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_dict(job)


@app.get("/v1/jobs/{job_id}/items/{index}/audio")
async def get_job_item_audio(job_id: str, index: int) -> Response:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if index < 0 or index >= len(job.items):
        raise HTTPException(status_code=404, detail="item index out of range")
    item = job.items[index]
    if item.status is not ItemStatus.DONE:
        raise HTTPException(status_code=409, detail=f"item not ready: {item.status.value}")
    wav_bytes = job_manager.audio_path(job_id, index).read_bytes()
    return Response(content=wav_bytes, media_type="audio/wav")


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_dict(job)
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `pytest tests/test_jobs_api.py -v`
Expected: PASS (9 passed)

- [ ] **Step 7: Run the full project test suite**

Run: `pytest -v`
Expected: everything passes (the pre-existing StarletteDeprecationWarning from importing TestClient is known and acceptable; it now originates from `tests/conftest.py`).

- [ ] **Step 8: Commit**

```bash
git add app/main.py tests/conftest.py tests/test_main.py tests/test_jobs_api.py
git commit -m "feat: add jobs + poll API endpoints"
```

---

## Task 4: Documentation

**Files:**
- Modify: `API.md`
- Modify: `README.md`

No TDD cycle (documentation only). Accuracy check instead of tests.

- [ ] **Step 1: Add a "Jobs API" section to `API.md`**

After the existing `POST /v1/tts/voice-design` section and before `GET /health`, insert a new section documenting all four endpoints with the exact request/response shapes from the Global Constraints and these curl examples (adjust wording to match the file's existing Vietnamese style):

````markdown
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

### `GET /v1/jobs/{job_id}/items/{index}/audio` — tải audio 1 item

```bash
curl http://127.0.0.1:8000/v1/jobs/j_a1b2c3d4e5f6/items/0/audio -o item0.wav
```

- Item `done` → `200` binary WAV. Tải được ngay khi item xong, không cần
  chờ cả job.
- Job không tồn tại hoặc index ngoài phạm vi → `404`.
- Item chưa xong (pending/running) hoặc failed/cancelled → `409`
  `{"detail": "item not ready: <status>"}`.

### `DELETE /v1/jobs/{job_id}` — hủy job

```bash
curl -X DELETE http://127.0.0.1:8000/v1/jobs/j_a1b2c3d4e5f6
```

- Item đang chạy trên GPU chạy nốt (kết quả vẫn tải được); item còn
  `pending` chuyển thành `cancelled`.
- Idempotent: hủy job đã xong/đã hủy → `200`, không đổi gì.
- Response: cùng shape với `GET /v1/jobs/{job_id}`.
````

Renumber the existing `GET /health` section heading accordingly (it becomes section 3).

- [ ] **Step 2: Update `README.md`**

In the `## Endpoints` section, after the `POST /v1/tts/voice-design` block, add:
```markdown
### Jobs API (bất đồng bộ, cho lô lớn)

- `POST /v1/jobs` — gửi 1 job chứa tối đa 1000 items, nhận `job_id` ngay (202)
- `GET /v1/jobs/{job_id}` — poll tiến độ per-item
- `GET /v1/jobs/{job_id}/items/{index}/audio` — tải WAV từng item khi xong
- `DELETE /v1/jobs/{job_id}` — hủy job

Chi tiết và curl mẫu: xem `API.md`. Kết quả job bị xóa mỗi lần server
khởi động lại; job không sống sót qua restart (client gửi lại).
```

In the `## Configuration` section's example block, add:
```bash
export QWEN_TTS_MAX_ITEMS_PER_JOB=1000
export QWEN_TTS_RESULTS_DIR=./results   # bị xóa sạch mỗi lần server khởi động
```

- [ ] **Step 3: Accuracy check**

Re-read both edited files against `app/main.py`'s actual routes/status codes and the Global Constraints (endpoint paths, 202/200/404/409, status value strings, env var names). Fix any mismatch found.

- [ ] **Step 4: Commit**

```bash
git add API.md README.md
git commit -m "docs: document jobs + poll API"
```

---

## Self-Review Notes

- **Spec coverage:** settings + submit schema → Task 1; JobManager (in-memory jobs, disk results, semaphore cap, per-item failure, cancel, shutdown, wipe) → Task 2; the four endpoints + lifespan wiring (wipe at startup, `job_manager.shutdown()` before `batch_worker.stop()`) + startup-wipe test → Task 3; docs → Task 4. The spec's "startup wipe test" is implemented BOTH as a direct unit test (Task 2, `test_wipe_results_dir_removes_stale_content`) and as a lifespan-level test (Task 3, `test_startup_wiped_stale_results` via the pre-seeded stale file in the conftest fixture).
- **Placeholder scan:** no TBD/TODO; every code step has complete, runnable code.
- **Type consistency:** `SubmitFn = Callable[[TTSRequest], Awaitable[bytes]]` in Task 2 matches `_job_submit`'s signature in Task 3 (`async def _job_submit(request: TTSRequest) -> bytes`). `JobManager(submit_fn, settings)` construction in Task 3 matches Task 2's `__init__`. `job_to_dict`/`ItemStatus` imports in Task 3's `main.py` match Task 2's exports. `JobSubmitRequest.items: list[TTSRequest]` (Task 1) matches `create_job(requests: list[TTSRequest])` (Task 2).
- **Known interaction risk, handled explicitly:** a second `TestClient` in a new test module would crash on the loop-bound `batch_worker` queue — Task 3 moves the fixture to a session-scoped `tests/conftest.py` FIRST (Step 1) and verifies existing tests still pass (Step 2) before adding the new test module.
- **`batch_worker`/`model.py` untouched:** verified — no task's file list includes `app/batcher.py` or `app/model.py`.
