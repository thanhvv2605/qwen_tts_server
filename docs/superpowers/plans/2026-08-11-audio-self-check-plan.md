# Audio Self-Check & Regenerate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect implausibly short/truncated TTS output server-side (word-count
vs. audio-duration heuristic) and regenerate it locally before replying, so
the client never has to detect the failure and retry over the network.

**Architecture:** `TTSModelService._generate_batch_sync` runs the existing
batch generation once, flags any item whose audio is too short for its word
count, and regenerates only the flagged subset (up to a retry cap) — still
inside the same threadpool-executor call, so nothing new touches the event
loop. Items still bad after retries become a per-item `Exception` value
rather than a raised exception, so `BatchWorker` must be extended to
distribute a **mix** of successes and per-item failures within one batch
(today it's all-or-nothing) — sibling requests in the same batch must not
be penalized by one unfixable item.

**Tech Stack:** Same as the existing project (Python 3.12, FastAPI, pytest +
pytest-asyncio, numpy/soundfile). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-11-audio-self-check-design.md`

## Global Constraints

- Detection: `abnormal = (len(wav) / sample_rate) < (len(text.split()) / MAX_PLAUSIBLE_WORDS_PER_SECOND)`
- `MAX_PLAUSIBLE_WORDS_PER_SECOND` default: `4.5` (env `QWEN_TTS_MAX_PLAUSIBLE_WORDS_PER_SECOND`)
- `AUDIO_SELF_CHECK_MAX_RETRIES` default: `2` (env `QWEN_TTS_AUDIO_SELF_CHECK_MAX_RETRIES`)
- On each retry, regenerate ONLY the still-abnormal subset, not the whole batch
- An item still abnormal after all retries becomes a per-item `Exception` in
  the returned list — it must NOT cause sibling items in the same batch to
  fail
- `BatchWorker`'s pre-existing whole-batch failure path (when `generate_fn`
  itself raises, e.g. a hard GPU crash) is unchanged — this is an
  **additional** failure path, not a replacement
- No change to the `POST /v1/tts/voice-design` request/response shape or to
  `app/main.py` — the existing exception → `500 {"detail": ...}` handling in
  `app/main.py:42-44` already covers a future raising with any `Exception`
- Given `app/batcher.py` is concurrency-sensitive code from a prior hardening
  pass, its task in this plan gets the same high-scrutiny review rigor
  (most-capable reviewer model, explicit focus on races/cancellation)

---

## Task 1: Settings

**Files:**
- Modify: `app/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: two new fields on `Settings`/`settings` — `max_plausible_words_per_second: float` (default `4.5`), `audio_self_check_max_retries: int` (default `2`).

- [ ] **Step 1: Write the failing test**

Update `tests/test_config.py` to:
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
    assert settings.max_plausible_words_per_second == 4.5
    assert settings.audio_self_check_max_retries == 2


def test_env_override(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_PORT", "9000")
    monkeypatch.setenv("QWEN_TTS_MAX_BATCH_SIZE", "8")
    settings = Settings(_env_file=None)
    assert settings.port == 9000
    assert settings.max_batch_size == 8


def test_env_override_audio_self_check_settings(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_MAX_PLAUSIBLE_WORDS_PER_SECOND", "3.0")
    monkeypatch.setenv("QWEN_TTS_AUDIO_SELF_CHECK_MAX_RETRIES", "5")
    settings = Settings(_env_file=None)
    assert settings.max_plausible_words_per_second == 3.0
    assert settings.audio_self_check_max_retries == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'max_plausible_words_per_second'`

- [ ] **Step 3: Write minimal implementation**

In `app/config.py`, add two fields to the `Settings` class (after `request_timeout_s`):
```python
    request_timeout_s: float = 120.0
    max_plausible_words_per_second: float = 4.5
    audio_self_check_max_retries: int = 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add audio self-check settings"
```

---

## Task 2: Detection heuristic

**Files:**
- Modify: `app/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Produces: `is_audio_abnormal(wav: np.ndarray, sample_rate: int, text: str, max_plausible_words_per_second: float) -> bool` — pure function, no GPU/model access. Task 4 will call this from `TTSModelService._generate_batch_sync`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py` (near the top, after the existing imports — no new imports needed, `numpy as np` is already imported):
```python
from app.model import is_audio_abnormal


def test_is_audio_abnormal_detects_truncated_audio():
    # The reported production case: 35 words compressed into 0.33s.
    wav = np.zeros(int(0.33 * 24000), dtype="float32")
    text = " ".join(["word"] * 35)
    assert is_audio_abnormal(wav, 24000, text, max_plausible_words_per_second=4.5) is True


def test_is_audio_abnormal_accepts_plausible_duration():
    # 35 words at ~2.5 words/sec = 14s, comfortably above the 4.5wps-implied minimum (~7.8s).
    wav = np.zeros(int(14.0 * 24000), dtype="float32")
    text = " ".join(["word"] * 35)
    assert is_audio_abnormal(wav, 24000, text, max_plausible_words_per_second=4.5) is False


def test_is_audio_abnormal_boundary_is_not_abnormal():
    # 9 words / 4.5 wps == exactly 2.0s - equal to the minimum should NOT be flagged.
    word_count = 9
    max_wps = 4.5
    wav = np.zeros(int((word_count / max_wps) * 24000), dtype="float32")
    text = " ".join(["word"] * word_count)
    assert is_audio_abnormal(wav, 24000, text, max_plausible_words_per_second=max_wps) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_audio_abnormal' from 'app.model'`

- [ ] **Step 3: Write minimal implementation**

In `app/model.py`, add this function (place it near `_wav_to_bytes`, before the `TTSModelService` class):
```python
def is_audio_abnormal(
    wav: np.ndarray, sample_rate: int, text: str, max_plausible_words_per_second: float
) -> bool:
    word_count = len(text.split())
    expected_min_duration_s = word_count / max_plausible_words_per_second
    actual_duration_s = len(wav) / sample_rate
    return actual_duration_s < expected_min_duration_s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (all tests in the file, including the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add app/model.py tests/test_model.py
git commit -m "feat: add audio duration self-check heuristic"
```

---

## Task 3: BatchWorker per-item failure isolation

**Files:**
- Modify: `app/batcher.py`
- Modify: `tests/test_batcher.py`

**Interfaces:**
- Consumes: `TTSRequest` from `app.schemas` (unchanged).
- Produces: `GenerateFn` type changes from `Callable[[Sequence[TTSRequest]], Awaitable[list[bytes]]]` to `Callable[[Sequence[TTSRequest]], Awaitable[list[bytes | Exception]]]`. `BatchWorker.submit(request) -> bytes` behavior unchanged from the caller's perspective (still raises the per-item exception if that item's slot was an `Exception`, still returns `bytes` on success) — only the internal distribution in `_dispatch` changes to support a per-item mix within one batch.

This task does NOT depend on Task 2 or Task 4 — it only changes how `_dispatch` interprets `generate_fn`'s return list, tested entirely with fakes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_batcher.py` (after the existing tests):
```python
async def test_mixed_success_and_failure_results_resolve_independently():
    async def mixed_generate(requests):
        results = []
        for r in requests:
            if r.text == "bad":
                results.append(RuntimeError("audio self-check failed"))
            else:
                results.append(f"audio-for-{r.text}".encode())
        return results

    worker = BatchWorker(generate_fn=mixed_generate, window_ms=100, max_batch_size=4)
    worker.start()
    try:
        task_good1 = asyncio.create_task(worker.submit(_req("good1")))
        task_bad = asyncio.create_task(worker.submit(_req("bad")))
        task_good2 = asyncio.create_task(worker.submit(_req("good2")))
        await asyncio.sleep(0.01)

        good1_result = await task_good1
        good2_result = await task_good2
        with pytest.raises(RuntimeError, match="audio self-check failed"):
            await task_bad
    finally:
        await worker.stop()

    assert good1_result == b"audio-for-good1"
    assert good2_result == b"audio-for-good2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_batcher.py -v`
Expected: FAIL. `_dispatch` currently calls `i.future.set_result(result)` for every item regardless of type, so the "bad" future resolves with a `RuntimeError` *object as its result value* instead of raising it — `await task_bad` returns the exception instance instead of raising, so `pytest.raises(...)` never triggers and the test fails with `Failed: DID NOT RAISE`.

- [ ] **Step 3: Write minimal implementation**

In `app/batcher.py`, change the `GenerateFn` type alias:
```python
GenerateFn = Callable[[Sequence[TTSRequest]], Awaitable[list[bytes | Exception]]]
```

Change `_dispatch`'s final distribution loop (the part after the `try/except` block) from:
```python
        for i, result in zip(batch, results):
            if not i.future.done():
                i.future.set_result(result)
```
to:
```python
        for i, result in zip(batch, results):
            if i.future.done():
                continue
            if isinstance(result, BaseException):
                i.future.set_exception(result)
            else:
                i.future.set_result(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_batcher.py -v`
Expected: PASS (all tests in the file, including the new one). Confirm specifically that `test_batch_error_propagates_to_all_pending_futures` and `test_mismatched_result_length_fails_batch_without_killing_worker` (the pre-existing whole-batch failure paths) still pass unchanged — they exercise `generate_fn` *raising*, a different code path from the per-item `Exception`-in-the-list path this task adds.

- [ ] **Step 5: Commit**

```bash
git add app/batcher.py tests/test_batcher.py
git commit -m "feat: support per-item failure isolation in BatchWorker dispatch"
```

---

## Task 4: Regenerate loop in TTSModelService

**Files:**
- Modify: `app/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `is_audio_abnormal` (Task 2, same file), `settings.max_plausible_words_per_second` / `settings.audio_self_check_max_retries` (Task 1), `BatchWorker`'s `GenerateFn` contract (Task 3) — this task is what actually PRODUCES the `list[bytes | Exception]` that contract expects.
- Produces: `TTSModelService.generate_batch` / `_generate_batch_sync` return type changes from `list[bytes]` to `list[bytes | Exception]`, one entry per input request, same order.

**Important — this task also fixes an existing test fixture.** The current
`_FakeModel` in `tests/test_model.py` (used by
`test_generate_batch_wires_requests_to_model`) always returns a fixed
2400-sample (0.1s) wav regardless of input text length. Once the self-check
runs, a 0.1s wav for any text longer than ~0 words will be flagged as
abnormal and trigger retries the fake doesn't expect, breaking that
pre-existing test. Fix `_FakeModel` to return a duration proportional to
word count (comfortably above the self-check threshold) as part of this
task's Step 3, so the existing test keeps testing what it always tested
(basic wiring) without tripping the new check.

- [ ] **Step 1: Write the failing tests**

In `tests/test_model.py`, replace the existing `_FakeModel` class with:
```python
class _FakeModel:
    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        # Duration well above the self-check threshold (2.5 words/sec is
        # slower than the 4.5 words/sec threshold, so this never trips
        # the self-check regardless of input text length).
        wavs = [
            np.zeros(max(int(len(t.split()) / 2.5 * 24000), 1), dtype="float32") for t in text
        ]
        return wavs, 24000
```
(`test_generate_batch_wires_requests_to_model`, which uses `_FakeModel`, is otherwise unchanged and should still pass once Task 4's implementation is in place — verify this alongside the new tests below, not as a separate step.)

Add these two new fake model classes and tests to `tests/test_model.py`:
```python
class _RecoversOnRetryModel:
    """The text "flaky" returns a truncated wav on its first appearance across
    all calls, then a normal-duration wav on every later call. Other texts
    always return a normal-duration wav. Records every call's text tuple."""

    def __init__(self):
        self.call_texts: list[tuple[str, ...]] = []
        self._seen_once: set[str] = set()

    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        self.call_texts.append(tuple(text))
        wavs = []
        for t in text:
            if t == "flaky" and t not in self._seen_once:
                self._seen_once.add(t)
                wavs.append(np.zeros(1, dtype="float32"))
            else:
                word_count = len(t.split())
                wavs.append(np.zeros(max(int(word_count / 2.5 * 24000), 1), dtype="float32"))
        return wavs, 24000


class _ControllableModel:
    """Texts in `always_bad_texts` return a truncated wav on every call,
    forever. Other texts always return a normal-duration wav. Records every
    call's text tuple."""

    def __init__(self, always_bad_texts: frozenset[str] = frozenset()):
        self._always_bad_texts = always_bad_texts
        self.call_texts: list[tuple[str, ...]] = []

    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        self.call_texts.append(tuple(text))
        wavs = []
        for t in text:
            if t in self._always_bad_texts:
                wavs.append(np.zeros(1, dtype="float32"))
            else:
                word_count = len(t.split())
                wavs.append(np.zeros(max(int(word_count / 2.5 * 24000), 1), dtype="float32"))
        return wavs, 24000


async def test_generate_batch_regenerates_abnormal_item_and_keeps_siblings():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    model = _RecoversOnRetryModel()
    service._model = model

    requests = [
        TTSRequest(text="good one", language="English", instruct="calm voice"),
        TTSRequest(text="flaky", language="English", instruct="calm voice"),
    ]

    results = await service.generate_batch(requests)

    assert len(results) == 2
    assert isinstance(results[0], bytes)
    assert isinstance(results[1], bytes)
    # First call: whole batch. Second call: only the still-abnormal subset ("flaky").
    assert model.call_texts == [("good one", "flaky"), ("flaky",)]


async def test_generate_batch_returns_exception_for_item_that_never_recovers():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    model = _ControllableModel(always_bad_texts=frozenset({"always broken"}))
    service._model = model

    requests = [
        TTSRequest(text="good one", language="English", instruct="calm voice"),
        TTSRequest(text="always broken", language="English", instruct="calm voice"),
    ]

    results = await service.generate_batch(requests)

    assert isinstance(results[0], bytes)
    assert isinstance(results[1], Exception)
    # 1 initial call + settings.audio_self_check_max_retries (2 by default) retries.
    assert len(model.call_texts) == 1 + settings.audio_self_check_max_retries
    for call in model.call_texts[1:]:
        assert call == ("always broken",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL for the 2 new tests (e.g. `assert len(results) == 2` type failures, since `_generate_batch_sync` doesn't self-check or retry yet — it will return the raw first-pass output, so `model.call_texts` will only have 1 entry, not 2 or 3, and `results[1]` will be `bytes` containing truncated audio instead of the expected retry/exception behavior). `test_generate_batch_wires_requests_to_model` should still PASS at this point (the `_FakeModel` fix alone doesn't require the retry logic).

- [ ] **Step 3: Write minimal implementation**

In `app/model.py`, replace `_generate_batch_sync` with:
```python
    def _generate_batch_sync(self, requests: list[TTSRequest]) -> list[bytes | Exception]:
        texts = [r.text for r in requests]
        languages = [r.language for r in requests]
        instructs = [r.instruct for r in requests]

        wavs, sr = self._model.generate_voice_design(
            text=texts,
            language=languages,
            instruct=instructs,
            max_new_tokens=self._settings.max_new_tokens,
        )

        max_wps = self._settings.max_plausible_words_per_second
        results: list[bytes | Exception | None] = [None] * len(requests)
        pending = [
            i
            for i, wav in enumerate(wavs)
            if is_audio_abnormal(wav, sr, texts[i], max_wps)
        ]
        for i, wav in enumerate(wavs):
            if i not in pending:
                results[i] = _wav_to_bytes(wav, sr)

        for _attempt in range(self._settings.audio_self_check_max_retries):
            if not pending:
                break
            retry_wavs, retry_sr = self._model.generate_voice_design(
                text=[texts[i] for i in pending],
                language=[languages[i] for i in pending],
                instruct=[instructs[i] for i in pending],
                max_new_tokens=self._settings.max_new_tokens,
            )
            still_pending = []
            for pos, i in enumerate(pending):
                wav = retry_wavs[pos]
                if is_audio_abnormal(wav, retry_sr, texts[i], max_wps):
                    still_pending.append(i)
                else:
                    results[i] = _wav_to_bytes(wav, retry_sr)
            pending = still_pending

        for i in pending:
            results[i] = RuntimeError(
                f"audio self-check failed after {self._settings.audio_self_check_max_retries} "
                f"retries for text {texts[i]!r}"
            )

        return results  # type: ignore[return-value]
```

Update `generate_batch`'s type annotation (the `async def` wrapper just above `_generate_batch_sync`) from `-> list[bytes]` to `-> list[bytes | Exception]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the full project test suite**

Run: `pytest -v`
Expected: all tests across every file pass, pristine output.

- [ ] **Step 6: Commit**

```bash
git add app/model.py tests/test_model.py
git commit -m "feat: regenerate abnormal audio in-place before returning"
```

---

## Self-Review Notes

- **Spec coverage:** detection heuristic → Task 2; regenerate-only-abnormal-subset loop, retry cap, per-item exception after exhausting retries → Task 4; `BatchWorker` per-item failure isolation (sibling items unaffected) → Task 3; new settings → Task 1. All spec sections covered. The spec's "no change to `app/main.py`" claim is preserved — no task touches `app/main.py`, since the existing `except Exception` → `500 {"detail": ...}` mapping in `app/main.py:42-44` already handles a future raising any exception, including the new per-item ones.
- **Placeholder scan:** no TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `GenerateFn` in Task 3 (`Callable[[Sequence[TTSRequest]], Awaitable[list[bytes | Exception]]]`) matches `TTSModelService.generate_batch`'s updated return type in Task 4 exactly — this is the interface Task 3 defines and Task 4 fulfills. `is_audio_abnormal`'s signature in Task 2 (`(wav, sample_rate, text, max_plausible_words_per_second)`) matches every call site added in Task 4.
- **Existing-test regression risk:** flagged explicitly in Task 4 — the pre-existing `_FakeModel` fixture would otherwise start failing once self-check logic lands, since its fixed 0.1s output would itself be flagged as abnormal for any real input text. Task 4 fixes the fixture as part of its own step rather than leaving it to break.
