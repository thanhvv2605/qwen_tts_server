# Qwen3-TTS VoiceDesign Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI REST server that generates speech with
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, batching concurrent requests
dynamically before sending them to the GPU.

**Architecture:** A single-process FastAPI app loads one `Qwen3TTSModel`
instance at startup (after checking free VRAM). Incoming requests are
queued and a single background `BatchWorker` task groups requests that
arrive within a short time window (or up to a max batch size) into one
call to `model.generate_voice_design(...)`, then distributes the
resulting WAV bytes back to each waiting request via `asyncio.Future`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, `qwen-tts` package
(`Qwen3TTSModel`), PyTorch, soundfile, pydantic-settings, pytest +
pytest-asyncio + httpx for tests.

**Spec:** `docs/superpowers/specs/2026-08-10-qwen-tts-voicedesign-server-design.md`

## Global Constraints

- `MODEL_ID` default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `DEVICE` default: `cuda:0`
- `HOST` default: `0.0.0.0`, `PORT` default: `8000`
- `MIN_FREE_VRAM_GB` default: `6.0`
- `BATCH_WINDOW_MS` default: `150`
- `MAX_BATCH_SIZE` default: `4`
- `MAX_NEW_TOKENS` default: `2048`
- `REQUEST_TIMEOUT_S` default: `120.0`
- `text` field: non-empty, max 2000 characters
- `instruct` field: non-empty (required for voice design to make sense)
- `language` field: one of `Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian`
- No authentication (local/internal use only)
- Exactly one background worker consumes the queue — GPU never receives two overlapping batches
- Skip FlashAttention 2; use Transformers' default attention implementation
- All env vars are prefixed `QWEN_TTS_` (e.g. `QWEN_TTS_PORT`)

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md`

**Interfaces:**
- Produces: installable env (`pip install -r requirements.txt`), `app` package importable, `pytest` runnable with zero errors.

- [ ] **Step 1: Create `requirements.txt`**

```text
qwen-tts>=0.1.0
fastapi>=0.115
uvicorn[standard]>=0.30
soundfile>=0.12
numpy>=1.26
pydantic-settings>=2.4
torch>=2.3
pytest>=8.0
pytest-asyncio>=0.24
httpx>=0.27
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 3: Create `.gitignore`**

```text
__pycache__/
*.pyc
.venv/
venv/
models/
*.wav
.pytest_cache/
.env
```

- [ ] **Step 4: Create empty package markers**

```bash
mkdir -p app tests scripts
touch app/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create `README.md` skeleton**

```markdown
# Qwen3-TTS VoiceDesign Server

Internal REST API server for `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.

## Setup

​```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
​```

## Run

​```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
​```

## Endpoints

- `POST /v1/tts/voice-design` — `{"text": "...", "language": "Auto", "instruct": "..."}` → returns `audio/wav`
- `GET /health` — server + GPU status
```

- [ ] **Step 6: Install dependencies and verify the environment**

Run: `conda create -n qwen3-tts python=3.12 -y && conda activate qwen3-tts && pip install -r requirements.txt`
Expected: install completes without errors (may take several minutes for `torch`/`qwen-tts`).

Run: `pytest`
Expected: `no tests ran` (exit code 0 — confirms pytest + asyncio plugin are wired correctly, no collection errors).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini .gitignore app/__init__.py tests/__init__.py README.md
git commit -m "chore: scaffold qwen-tts server project"
```

---

## Task 2: Settings

**Files:**
- Create: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` class and `settings` singleton in `app/config.py`, fields: `model_id: str`, `device: str`, `host: str`, `port: int`, `min_free_vram_gb: float`, `batch_window_ms: int`, `max_batch_size: int`, `max_new_tokens: int`, `request_timeout_s: float`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from app.config import Settings


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert settings.device == "cuda:0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.min_free_vram_gb == 6.0
    assert settings.batch_window_ms == 150
    assert settings.max_batch_size == 4
    assert settings.max_new_tokens == 2048
    assert settings.request_timeout_s == 120.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_PORT", "9000")
    monkeypatch.setenv("QWEN_TTS_MAX_BATCH_SIZE", "8")
    settings = Settings(_env_file=None)
    assert settings.port == 9000
    assert settings.max_batch_size == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Write minimal implementation**

`app/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN_TTS_")

    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    min_free_vram_gb: float = 6.0
    batch_window_ms: int = 150
    max_batch_size: int = 4
    max_new_tokens: int = 2048
    request_timeout_s: float = 120.0


settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add server settings"
```

---

## Task 3: Request schema & validation

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TTSRequest(BaseModel)` with fields `text: str`, `language: str = "Auto"`, `instruct: str`, raising `pydantic.ValidationError` on invalid input; `SUPPORTED_LANGUAGES: set[str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from app.schemas import TTSRequest


def test_valid_request():
    req = TTSRequest(text="Hello", language="English", instruct="calm voice")
    assert req.text == "Hello"
    assert req.language == "English"
    assert req.instruct == "calm voice"


def test_default_language_is_auto():
    req = TTSRequest(text="Hello", instruct="calm voice")
    assert req.language == "Auto"


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="   ", language="English", instruct="calm voice")


def test_text_too_long_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="x" * 2001, language="English", instruct="calm voice")


def test_empty_instruct_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", language="English", instruct="  ")


def test_unsupported_language_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", language="Klingon", instruct="calm voice")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 3: Write minimal implementation**

`app/schemas.py`:
```python
from pydantic import BaseModel, field_validator

SUPPORTED_LANGUAGES = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}


class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    instruct: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        if len(v) > 2000:
            raise ValueError("text must be at most 2000 characters")
        return v

    @field_validator("instruct")
    @classmethod
    def instruct_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("instruct must not be empty")
        return v

    @field_validator("language")
    @classmethod
    def language_supported(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add TTSRequest schema with validation"
```

---

## Task 4: Batching worker

**Files:**
- Create: `app/batcher.py`
- Test: `tests/test_batcher.py`

**Interfaces:**
- Consumes: `TTSRequest` from `app.schemas` (Task 3).
- Produces: `BatchWorker(generate_fn, window_ms, max_batch_size)` with:
  - `async def submit(request: TTSRequest) -> bytes` — enqueue and await the result
  - `def start() -> None` — start the background loop
  - `async def stop() -> None` — cancel and await the background loop
  - `def queue_depth() -> int` — number of items currently queued (not yet dispatched)
  - `generate_fn` type: `Callable[[list[TTSRequest]], Awaitable[list[bytes]]]`, called with one batch, must return a list of `bytes` the same length and order as the input list.

- [ ] **Step 1: Write the failing test**

`tests/test_batcher.py`:
```python
import asyncio

import pytest

from app.batcher import BatchWorker
from app.schemas import TTSRequest


def _req(text: str) -> TTSRequest:
    return TTSRequest(text=text, language="English", instruct="calm voice")


async def test_single_request_is_processed():
    calls = []

    async def fake_generate(requests):
        calls.append(list(requests))
        return [f"audio-for-{r.text}".encode() for r in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=50, max_batch_size=4)
    worker.start()
    try:
        result = await worker.submit(_req("hello"))
    finally:
        await worker.stop()

    assert result == b"audio-for-hello"
    assert calls == [[_req("hello")]]


async def test_requests_within_window_are_batched_together():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=100, max_batch_size=4)
    worker.start()
    try:
        task_a = asyncio.create_task(worker.submit(_req("a")))
        await asyncio.sleep(0.02)
        task_b = asyncio.create_task(worker.submit(_req("b")))
        await task_a
        await task_b
    finally:
        await worker.stop()

    assert calls == [2]


async def test_requests_outside_window_are_separate_batches():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=30, max_batch_size=4)
    worker.start()
    try:
        await worker.submit(_req("a"))
        await worker.submit(_req("b"))
    finally:
        await worker.stop()

    assert calls == [1, 1]


async def test_batch_caps_at_max_batch_size():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=200, max_batch_size=2)
    worker.start()
    try:
        tasks = [asyncio.create_task(worker.submit(_req(str(i)))) for i in range(3)]
        await asyncio.gather(*tasks)
    finally:
        await worker.stop()

    assert calls == [2, 1]


async def test_batch_error_propagates_to_all_pending_futures():
    async def failing_generate(requests):
        raise RuntimeError("boom")

    worker = BatchWorker(generate_fn=failing_generate, window_ms=50, max_batch_size=4)
    worker.start()
    try:
        task_a = asyncio.create_task(worker.submit(_req("a")))
        await asyncio.sleep(0.01)
        task_b = asyncio.create_task(worker.submit(_req("b")))

        with pytest.raises(RuntimeError, match="boom"):
            await task_a
        with pytest.raises(RuntimeError, match="boom"):
            await task_b
    finally:
        await worker.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.batcher'`

- [ ] **Step 3: Write minimal implementation**

`app/batcher.py`:
```python
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

GenerateFn = Callable[[Sequence[TTSRequest]], Awaitable[list[bytes]]]


@dataclass
class _QueueItem:
    request: TTSRequest
    future: "asyncio.Future[bytes]"


class BatchWorker:
    def __init__(self, generate_fn: GenerateFn, window_ms: int, max_batch_size: int) -> None:
        self._generate_fn = generate_fn
        self._window_s = window_ms / 1000
        self._max_batch_size = max_batch_size
        self._queue: "asyncio.Queue[_QueueItem]" = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def submit(self, request: TTSRequest) -> bytes:
        future: "asyncio.Future[bytes]" = asyncio.get_running_loop().create_future()
        await self._queue.put(_QueueItem(request, future))
        return await future

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            batch = [item]
            deadline = time.monotonic() + self._window_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(next_item)
            await self._dispatch(batch)

    async def _dispatch(self, batch: list[_QueueItem]) -> None:
        requests = [i.request for i in batch]
        try:
            results = await self._generate_fn(requests)
        except Exception as exc:  # noqa: BLE001 - propagate to callers via their Future
            logger.exception("Batch of %d request(s) failed", len(batch))
            for i in batch:
                if not i.future.done():
                    i.future.set_exception(exc)
            return
        for i, result in zip(batch, results):
            if not i.future.done():
                i.future.set_result(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batcher.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/batcher.py tests/test_batcher.py
git commit -m "feat: add dynamic batching worker"
```

---

## Task 5: Model service (VRAM check, WAV encoding, generation wiring)

**Files:**
- Create: `app/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Settings` from `app.config` (Task 2), `TTSRequest` from `app.schemas` (Task 3).
- Produces:
  - `check_vram(device: str, min_free_gb: float) -> float` — returns free VRAM in GB, logs a WARNING if below `min_free_gb`, raises `RuntimeError` if CUDA unavailable.
  - `TTSModelService(settings: Settings)` with `load() -> None`, `is_loaded() -> bool`, `async def generate_batch(requests: Sequence[TTSRequest]) -> list[bytes]` (matches `BatchWorker`'s `GenerateFn` type from Task 4).

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import io
import logging

import numpy as np
import pytest
import soundfile as sf
import torch

from app.config import Settings
from app.model import TTSModelService, _wav_to_bytes, check_vram
from app.schemas import TTSRequest


def test_check_vram_returns_free_gb(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (12 * 1024**3, 24 * 1024**3))
    free_gb = check_vram("cuda:0", min_free_gb=6.0)
    assert free_gb == pytest.approx(12.0, abs=0.01)


def test_check_vram_warns_when_low(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (2 * 1024**3, 24 * 1024**3))
    with caplog.at_level(logging.WARNING):
        check_vram("cuda:0", min_free_gb=6.0)
    assert "VRAM" in caplog.text


def test_check_vram_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        check_vram("cuda:0", min_free_gb=6.0)


def test_wav_to_bytes_roundtrip():
    wav = np.zeros(4800, dtype="float32")
    data = _wav_to_bytes(wav, 24000)
    read_wav, sr = sf.read(io.BytesIO(data))
    assert sr == 24000
    assert len(read_wav) == 4800


class _FakeModel:
    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        wavs = [np.zeros(2400, dtype="float32") for _ in text]
        return wavs, 24000


async def test_generate_batch_wires_requests_to_model():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    service._model = _FakeModel()

    requests = [
        TTSRequest(text="hello", language="English", instruct="calm voice"),
        TTSRequest(text="world", language="English", instruct="excited voice"),
    ]

    results = await service.generate_batch(requests)

    assert len(results) == 2
    for wav_bytes in results:
        data, sr = sf.read(io.BytesIO(wav_bytes))
        assert sr == 24000


def test_is_loaded_false_before_load():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    assert service.is_loaded() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.model'`

- [ ] **Step 3: Write minimal implementation**

`app/model.py`:
```python
import asyncio
import io
import logging
import subprocess
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

from app.config import Settings
from app.schemas import TTSRequest

logger = logging.getLogger(__name__)


def check_vram(device: str, min_free_gb: float) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine")
    index = torch.device(device).index or 0
    free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
    free_gb = free_bytes / (1024**3)
    if free_gb < min_free_gb:
        logger.warning(
            "Only %.1fGB VRAM free on %s (< %.1fGB threshold). Processes using this GPU:\n%s",
            free_gb,
            device,
            min_free_gb,
            _nvidia_smi_processes(),
        )
    return free_gb


def _nvidia_smi_processes() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic only
        return f"(could not run nvidia-smi: {exc})"


def _wav_to_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    return buffer.getvalue()


class TTSModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None

    def load(self) -> None:
        check_vram(self._settings.device, self._settings.min_free_vram_gb)
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            self._settings.model_id,
            device_map=self._settings.device,
            dtype=torch.bfloat16,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    async def generate_batch(self, requests: Sequence[TTSRequest]) -> list[bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_batch_sync, list(requests))

    def _generate_batch_sync(self, requests: list[TTSRequest]) -> list[bytes]:
        wavs, sr = self._model.generate_voice_design(
            text=[r.text for r in requests],
            language=[r.language for r in requests],
            instruct=[r.instruct for r in requests],
            max_new_tokens=self._settings.max_new_tokens,
        )
        return [_wav_to_bytes(wav, sr) for wav in wavs]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/model.py tests/test_model.py
git commit -m "feat: add TTS model service with VRAM check and WAV encoding"
```

---

## Task 6: FastAPI app

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `settings` from `app.config` (Task 2), `TTSRequest` from `app.schemas` (Task 3), `BatchWorker` from `app.batcher` (Task 4), `TTSModelService`/`check_vram` from `app.model` (Task 5).
- Produces: `app` (FastAPI instance), module-level `model_service: TTSModelService`, `batch_worker: BatchWorker` — both accessible as `app.main.model_service` / `app.main.batch_worker` for test monkeypatching.

- [ ] **Step 1: Write the failing test**

`tests/test_main.py`:
```python
import io

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from app import main as main_module


def _fake_wav_bytes() -> bytes:
    wav = np.zeros(2400, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module.model_service, "load", lambda: None)
    monkeypatch.setattr(main_module.model_service, "is_loaded", lambda: True)
    monkeypatch.setattr("app.model.check_vram", lambda device, min_free_gb: 20.0)

    async def fake_generate_fn(requests):
        return [_fake_wav_bytes() for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", fake_generate_fn)

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_voice_design_returns_wav(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "instruct": "calm male voice"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 24000


def test_voice_design_rejects_empty_text(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "", "language": "English", "instruct": "calm"},
    )
    assert resp.status_code == 422


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["vram_free_gb"] == 20.0
    assert body["queue_depth"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Write minimal implementation**

`app/main.py`:
```python
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from app.batcher import BatchWorker
from app.config import settings
from app.model import TTSModelService, check_vram
from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

model_service = TTSModelService(settings)
batch_worker = BatchWorker(
    generate_fn=model_service.generate_batch,
    window_ms=settings.batch_window_ms,
    max_batch_size=settings.max_batch_size,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    batch_worker.start()
    yield
    await batch_worker.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/tts/voice-design")
async def voice_design(request: TTSRequest) -> Response:
    try:
        wav_bytes = await asyncio.wait_for(
            batch_worker.submit(request), timeout=settings.request_timeout_s
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Timed out waiting for GPU queue") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 500
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
async def health() -> dict:
    vram_free_gb = None
    if model_service.is_loaded():
        vram_free_gb = check_vram(settings.device, settings.min_free_vram_gb)
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "vram_free_gb": vram_free_gb,
        "queue_depth": batch_worker.queue_depth(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests from Tasks 2-6 PASS (around 22 passed), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add FastAPI app with voice-design and health endpoints"
```

---

## Task 7: Smoke test script and final docs

**Files:**
- Create: `scripts/smoke_test.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: a running server (real GPU + real model, started manually — not part of the automated test suite).
- Produces: a manual verification script an operator runs after `uvicorn` is up.

- [ ] **Step 1: Write the smoke test script**

`scripts/smoke_test.py`:
```python
"""Manual smoke test: run this against a live server after `uvicorn app.main:app` is up.

Usage:
    python scripts/smoke_test.py [base_url]
"""

import io
import sys

import httpx
import soundfile as sf

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def main() -> None:
    health_resp = httpx.get(f"{BASE_URL}/health", timeout=10)
    health_resp.raise_for_status()
    print("GET /health ->", health_resp.json())
    assert health_resp.json()["model_loaded"] is True, "model did not load"

    tts_resp = httpx.post(
        f"{BASE_URL}/v1/tts/voice-design",
        json={
            "text": "Xin chao, day la mot bai kiem tra.",
            "language": "Auto",
            "instruct": "Giong nu tre, vui ve, toc do noi vua phai.",
        },
        timeout=120,
    )
    tts_resp.raise_for_status()
    assert tts_resp.headers["content-type"] == "audio/wav"

    data, sr = sf.read(io.BytesIO(tts_resp.content))
    print(f"POST /v1/tts/voice-design -> {len(data)} samples at {sr}Hz")
    assert len(data) > 0, "empty audio returned"

    with open("smoke_test_output.wav", "wb") as f:
        f.write(tts_resp.content)
    print("Saved smoke_test_output.wav — listen to it to confirm audio quality.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against a live server to verify end-to-end behavior**

Run:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 5   # wait for model to load — first run also downloads ~4GB from HuggingFace
python scripts/smoke_test.py
```
Expected: both requests succeed, `smoke_test_output.wav` is created, printed sample count > 0. Manually play `smoke_test_output.wav` to confirm it sounds like the requested voice description.

- [ ] **Step 3: Update `README.md` with the finished endpoint contract and config table**

Replace the `## Endpoints` section of `README.md` with:
```markdown
## Endpoints

### `POST /v1/tts/voice-design`

Request body:
​```json
{
  "text": "Xin chao, hom nay troi dep qua.",
  "language": "Auto",
  "instruct": "Giong nu tre, vui ve, toc do noi nhanh."
}
​```
- `language` one of: Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Response: `audio/wav` binary (200), or `422` (invalid input), `504` (queue timeout), `500` (generation error)

### `GET /health`

Returns `{"status", "model_loaded", "vram_free_gb", "queue_depth"}`.

## Configuration

All settings are environment variables prefixed `QWEN_TTS_` (see `app/config.py` for defaults), e.g.:

​```bash
export QWEN_TTS_PORT=8080
export QWEN_TTS_MAX_BATCH_SIZE=8
export QWEN_TTS_MODEL_ID=./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign  # use a local path to skip re-downloading
​```

## Manual verification

After starting the server, run `python scripts/smoke_test.py` to confirm the model loaded and a real WAV file comes back.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py README.md
git commit -m "docs: add smoke test script and finish README"
```

---

## Self-Review Notes

- **Spec coverage:** VRAM check → Task 5 (`check_vram`); dynamic batching (window/max-size/single-worker) → Task 4; validation rules (text/instruct non-empty, length, language whitelist) → Task 3; `POST /v1/tts/voice-design` + `GET /health` + error codes (422/500/504) → Task 6; setup/conda/deps → Task 1; manual e2e verification → Task 7. All spec sections are covered.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `GenerateFn` in Task 4 (`Callable[[Sequence[TTSRequest]], Awaitable[list[bytes]]]`) matches `TTSModelService.generate_batch`'s signature in Task 5, which is what `main.py` passes into `BatchWorker(generate_fn=model_service.generate_batch, ...)` in Task 6.
