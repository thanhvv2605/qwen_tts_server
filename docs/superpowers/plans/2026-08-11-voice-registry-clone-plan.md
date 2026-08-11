# Voice Registry + Clone Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persistent voice registry (register a reference WAV+transcript once
→ stable `voice_id`) plus clone-based generation via the Qwen3-TTS Base
model, so every audio generated with a `voice_id` comes out in the same
voice — on the sync endpoint and (critically) on 578-item jobs.

**Architecture:** A new `VoiceRegistry` (`app/voices.py`) stores reference
audio+text under `VOICES_DIR/{voice_id}/` (persistent, scanned at startup —
NOT wiped, unlike `RESULTS_DIR`). `TTSModelService` loads a second
checkpoint (`Qwen3-TTS-12Hz-1.7B-Base`) alongside VoiceDesign, caches
lazily-built clone prompts per voice, and its `_generate_batch_sync`
becomes a splitter: design items (instruct) → VoiceDesign; clone items
(voice_id) grouped per voice → Base model, one call per group. The existing
audio self-check + retry loop is refactored into a group-generic helper
used by BOTH paths. `TTSRequest` gains `voice_id` with exactly-one-of
`instruct`/`voice_id` validation — jobs inherit it automatically.
`BatchWorker` and `JobManager` are UNCHANGED.

**Tech Stack:** existing stack; `python-multipart` added to requirements
(needed by FastAPI form/file upload; already present in the env).

**Spec:** `docs/superpowers/specs/2026-08-11-voice-registry-clone-design.md`

## Global Constraints

- New settings (env prefix `QWEN_TTS_`): `clone_model_id` default `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"`, `voices_dir` default `"./voices"`, `voice_clone_enabled` default `True`
- `TTSRequest`: exactly ONE of `instruct` / `voice_id` must be provided (both → 422, neither → 422); provided-but-blank values also 422; `text`/`language` rules unchanged
- Voice name/id: client-chosen slug matching `[a-z0-9_-]{1,64}`; duplicate registration → 409; invalid slug/audio/duration/ref_text → 422; registry endpoints → 503 when `voice_clone_enabled` is false
- Reference audio: must decode via `soundfile`; duration must be within 0.5s–60s
- Storage: `VOICES_DIR/{voice_id}/ref.wav` + `ref.txt`; persistent across restarts (startup scan, skip-with-warning on malformed entries); `voices/` in `.gitignore`
- Clone prompts built lazily on first use per voice (inside the generation executor thread) via `create_voice_clone_prompt(ref_audio=<path>, ref_text=..., x_vector_only_mode=False)`, cached in memory; cache invalidated on voice deletion
- Mixed batches split: design items in ONE `generate_voice_design` call; clone items grouped by `voice_id`, ONE `generate_voice_clone(text=[...], language=[...], voice_clone_prompt=...)` call per group (the Base model takes no `instruct`)
- Audio self-check (threshold/retries/counters/log messages) applies identically to both paths, retrying only the abnormal subset within each group
- Failure isolation: unknown/disabled/failed-prompt `voice_id` → per-item `Exception`s for that group ONLY (message `"unknown voice_id: <id>"` / `"voice cloning is disabled"`); a group's initial-generation failure fails only that group. (This upgrades today's behavior where a whole-batch initial failure raised out of `_generate_batch_sync` — one existing test changes accordingly, called out in Task 3.)
- `/health` gains `clone_model_loaded: bool`
- Endpoints: `POST /v1/voices` (multipart `name`+`ref_text`+`ref_audio`) → 201 `{voice_id, duration_s}`; `GET /v1/voices` → 200 `{voices: [{voice_id, duration_s, ref_text}]}`; `DELETE /v1/voices/{voice_id}` → 200 `{deleted}` / 404
- Registration/deletion disk I/O runs via `run_in_executor` (never block the event loop — established convention in this codebase)
- `app/batcher.py` and `app/jobs.py` are NOT modified

---

## Task 1: Settings and TTSRequest voice_id validation

**Files:**
- Modify: `app/config.py`
- Modify: `app/schemas.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Modify: `tests/test_config.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**
- Produces: `settings.clone_model_id: str`, `settings.voices_dir: str`, `settings.voice_clone_enabled: bool`; `TTSRequest` with `instruct: str | None = None` and `voice_id: str | None = None` under exactly-one-of validation. Existing constructions `TTSRequest(text=..., language=..., instruct=...)` (used across all test helpers) remain valid unchanged.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, extend `test_default_settings` with:
```python
    assert settings.clone_model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert settings.voices_dir == "./voices"
    assert settings.voice_clone_enabled is True
```
and add:
```python
def test_env_override_voice_clone_settings(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_CLONE_MODEL_ID", "./models/base")
    monkeypatch.setenv("QWEN_TTS_VOICES_DIR", "/tmp/tts-voices")
    monkeypatch.setenv("QWEN_TTS_VOICE_CLONE_ENABLED", "false")
    settings = Settings(_env_file=None)
    assert settings.clone_model_id == "./models/base"
    assert settings.voices_dir == "/tmp/tts-voices"
    assert settings.voice_clone_enabled is False
```

In `tests/test_schemas.py`, add:
```python
def test_voice_id_alone_is_valid():
    req = TTSRequest(text="hello", language="English", voice_id="astronomy_male_en")
    assert req.voice_id == "astronomy_male_en"
    assert req.instruct is None


def test_instruct_alone_is_valid_unchanged():
    req = TTSRequest(text="hello", language="English", instruct="calm voice")
    assert req.instruct == "calm voice"
    assert req.voice_id is None


def test_both_instruct_and_voice_id_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", instruct="calm voice", voice_id="astronomy_male_en")


def test_neither_instruct_nor_voice_id_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", language="English")


def test_blank_voice_id_rejected():
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", voice_id="   ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_schemas.py -v`
Expected: FAIL — `AttributeError` on new settings; `ValidationError` NOT raised / unexpected-keyword errors on the schema tests. Note `test_neither_instruct_nor_voice_id_rejected` may PASS vacuously today (missing `instruct` is already an error) — that's fine; it pins the future contract.

- [ ] **Step 3: Write minimal implementation**

`app/config.py` — add after `results_dir`:
```python
    results_dir: str = "./results"
    clone_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    voices_dir: str = "./voices"
    voice_clone_enabled: bool = True
```

`app/schemas.py` — change the import line and the `TTSRequest` class:
```python
from pydantic import BaseModel, Field, field_validator, model_validator
```
```python
class TTSRequest(BaseModel):
    text: str
    language: str = "Auto"
    instruct: str | None = None
    voice_id: str | None = None

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
    def instruct_not_empty(cls, v: "str | None") -> "str | None":
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("instruct must not be empty when provided")
        return v

    @field_validator("voice_id")
    @classmethod
    def voice_id_not_empty(cls, v: "str | None") -> "str | None":
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("voice_id must not be empty when provided")
        return v

    @field_validator("language")
    @classmethod
    def language_supported(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(SUPPORTED_LANGUAGES)}")
        return v

    @model_validator(mode="after")
    def exactly_one_of_instruct_or_voice_id(self) -> "TTSRequest":
        if (self.instruct is None) == (self.voice_id is None):
            raise ValueError("exactly one of 'instruct' or 'voice_id' must be provided")
        return self
```

`.gitignore` — append:
```text
voices/
```

`requirements.txt` — append:
```text
python-multipart>=0.0.9
```

- [ ] **Step 4: Run the FULL suite**

Run: `pytest -v`
Expected: everything passes — the existing `test_empty_instruct_rejected` (explicit blank instruct) still fails validation via the adapted validator; all `_req()` helpers across test files pass `instruct` explicitly and remain valid.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/schemas.py .gitignore requirements.txt tests/test_config.py tests/test_schemas.py
git commit -m "feat: add voice-clone settings and TTSRequest voice_id"
```

---

## Task 2: VoiceRegistry

**Files:**
- Create: `app/voices.py`
- Create: `tests/test_voices.py`

**Interfaces:**
- Consumes: `Settings` (reads `settings.voices_dir` dynamically per call, same pattern as `JobManager`).
- Produces (Tasks 3–4 depend on these):
  - Exceptions: `VoiceRegistryError(Exception)`, `DuplicateVoiceError(VoiceRegistryError)`, `InvalidVoiceError(VoiceRegistryError)`
  - `VoiceInfo` dataclass: `voice_id: str`, `ref_text: str`, `duration_s: float`, `wav_path: Path`
  - `VoiceRegistry(settings)` with `scan() -> None`, `register(name, audio_bytes, ref_text) -> VoiceInfo`, `get(voice_id) -> VoiceInfo | None`, `list_voices() -> list[VoiceInfo]` (sorted by id), `delete(voice_id) -> bool`
  - Constants `MIN_REF_DURATION_S = 0.5`, `MAX_REF_DURATION_S = 60.0`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voices.py`:
```python
import io

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.voices import (
    DuplicateVoiceError,
    InvalidVoiceError,
    VoiceRegistry,
)


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, voices_dir=str(tmp_path / "voices"))


def _wav_bytes(duration_s: float = 2.0, samplerate: int = 24000) -> bytes:
    wav = np.zeros(int(duration_s * samplerate), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, samplerate, format="WAV")
    return buf.getvalue()


def test_register_get_list_delete_roundtrip(tmp_path):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()

    info = reg.register("astro_male_en", _wav_bytes(2.0), "Hello there.")
    assert info.voice_id == "astro_male_en"
    assert info.duration_s == pytest.approx(2.0, abs=0.01)
    assert info.wav_path.exists()
    assert info.wav_path.with_name("ref.txt").read_text(encoding="utf-8") == "Hello there."

    assert reg.get("astro_male_en") is info
    assert [v.voice_id for v in reg.list_voices()] == ["astro_male_en"]

    assert reg.delete("astro_male_en") is True
    assert reg.get("astro_male_en") is None
    assert not info.wav_path.exists()
    assert reg.delete("astro_male_en") is False


def test_register_duplicate_name_rejected(tmp_path):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()
    reg.register("voice_a", _wav_bytes(), "text")
    with pytest.raises(DuplicateVoiceError):
        reg.register("voice_a", _wav_bytes(), "other text")


@pytest.mark.parametrize("bad_name", ["", "UPPER", "has space", "a" * 65, "việt"])
def test_register_invalid_name_rejected(tmp_path, bad_name):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()
    with pytest.raises(InvalidVoiceError):
        reg.register(bad_name, _wav_bytes(), "text")


def test_register_undecodable_audio_rejected(tmp_path):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()
    with pytest.raises(InvalidVoiceError):
        reg.register("voice_a", b"this is not audio", "text")


@pytest.mark.parametrize("duration_s", [0.2, 90.0])
def test_register_out_of_range_duration_rejected(tmp_path, duration_s):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()
    with pytest.raises(InvalidVoiceError):
        reg.register("voice_a", _wav_bytes(duration_s), "text")


def test_register_empty_ref_text_rejected(tmp_path):
    reg = VoiceRegistry(_settings(tmp_path))
    reg.scan()
    with pytest.raises(InvalidVoiceError):
        reg.register("voice_a", _wav_bytes(), "   ")


def test_scan_recovers_previously_registered_voices(tmp_path):
    settings = _settings(tmp_path)
    reg1 = VoiceRegistry(settings)
    reg1.scan()
    reg1.register("voice_a", _wav_bytes(3.0), "persisted text")

    reg2 = VoiceRegistry(settings)
    reg2.scan()
    info = reg2.get("voice_a")
    assert info is not None
    assert info.ref_text == "persisted text"
    assert info.duration_s == pytest.approx(3.0, abs=0.01)


def test_scan_skips_malformed_entries(tmp_path, caplog):
    settings = _settings(tmp_path)
    root = tmp_path / "voices"
    (root / "broken").mkdir(parents=True)  # dir without ref.wav/ref.txt
    (root / "not_audio").mkdir()
    (root / "not_audio" / "ref.wav").write_bytes(b"garbage")
    (root / "not_audio" / "ref.txt").write_text("t", encoding="utf-8")

    reg = VoiceRegistry(settings)
    reg.scan()
    assert reg.list_voices() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.voices'`

- [ ] **Step 3: Write minimal implementation**

Create `app/voices.py`:
```python
import logging
import re
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import soundfile as sf

from app.config import Settings

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
MIN_REF_DURATION_S = 0.5
MAX_REF_DURATION_S = 60.0


class VoiceRegistryError(Exception):
    pass


class DuplicateVoiceError(VoiceRegistryError):
    pass


class InvalidVoiceError(VoiceRegistryError):
    pass


@dataclass
class VoiceInfo:
    voice_id: str
    ref_text: str
    duration_s: float
    wav_path: Path


class VoiceRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voices: dict[str, VoiceInfo] = {}

    def _root(self) -> Path:
        return Path(self._settings.voices_dir)

    def scan(self) -> None:
        self._voices = {}
        root = self._root()
        root.mkdir(parents=True, exist_ok=True)
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            wav_path = entry / "ref.wav"
            txt_path = entry / "ref.txt"
            if not wav_path.exists() or not txt_path.exists():
                logger.warning(
                    "skipping malformed voice entry %s (missing ref.wav/ref.txt)", entry
                )
                continue
            try:
                info = sf.info(str(wav_path))
                duration_s = info.frames / info.samplerate
                ref_text = txt_path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001 - one bad entry must not break boot
                logger.warning("skipping unreadable voice entry %s", entry, exc_info=True)
                continue
            self._voices[entry.name] = VoiceInfo(entry.name, ref_text, duration_s, wav_path)
        logger.info("voice registry loaded %d voice(s) from %s", len(self._voices), root)

    def register(self, name: str, audio_bytes: bytes, ref_text: str) -> VoiceInfo:
        if not _NAME_RE.match(name or ""):
            raise InvalidVoiceError("name must match [a-z0-9_-]{1,64}")
        if name in self._voices:
            raise DuplicateVoiceError(f"voice {name!r} already exists")
        ref_text = (ref_text or "").strip()
        if not ref_text:
            raise InvalidVoiceError("ref_text must not be empty")
        try:
            data, samplerate = sf.read(BytesIO(audio_bytes))
        except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
            raise InvalidVoiceError(f"ref_audio is not decodable audio: {exc}") from exc
        duration_s = len(data) / samplerate
        if not (MIN_REF_DURATION_S <= duration_s <= MAX_REF_DURATION_S):
            raise InvalidVoiceError(
                f"ref_audio must be between {MIN_REF_DURATION_S}s and "
                f"{MAX_REF_DURATION_S}s, got {duration_s:.2f}s"
            )
        voice_dir = self._root() / name
        voice_dir.mkdir(parents=True, exist_ok=True)
        wav_path = voice_dir / "ref.wav"
        wav_path.write_bytes(audio_bytes)
        (voice_dir / "ref.txt").write_text(ref_text, encoding="utf-8")
        info = VoiceInfo(name, ref_text, duration_s, wav_path)
        self._voices[name] = info
        return info

    def get(self, voice_id: str) -> "VoiceInfo | None":
        return self._voices.get(voice_id)

    def list_voices(self) -> list[VoiceInfo]:
        return sorted(self._voices.values(), key=lambda v: v.voice_id)

    def delete(self, voice_id: str) -> bool:
        info = self._voices.pop(voice_id, None)
        if info is None:
            return False
        shutil.rmtree(self._root() / voice_id, ignore_errors=True)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voices.py -v`
Expected: PASS (12 passed — 8 functions, two parametrized)

- [ ] **Step 5: Commit**

```bash
git add app/voices.py tests/test_voices.py
git commit -m "feat: add persistent VoiceRegistry"
```

---

## Task 3: Model service — dual model, batch splitting, clone prompts

**Files:**
- Modify: `app/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `VoiceRegistry`/`VoiceInfo` from `app.voices` (Task 2); `TTSRequest.voice_id` (Task 1).
- Produces:
  - `TTSModelService(settings, voice_registry=None)` — second positional param, default `None` so existing constructions keep working
  - `clone_is_loaded() -> bool`
  - `invalidate_clone_prompt(voice_id: str) -> None`
  - `load()` now loads the Base checkpoint too when `settings.voice_clone_enabled`
  - `_generate_batch_sync` splits design/clone groups; the self-check loop moves into a group-generic `_run_group(requests, indices, gen_fn, results)` used by both paths
- **BEHAVIOR CHANGE (intentional, per spec):** an initial-generation failure now produces per-item `Exception`s for the failing GROUP instead of raising out of `_generate_batch_sync` (which previously failed the whole batch via the batcher's catch). One existing test asserts the old behavior and must be UPDATED: `test_generate_batch_raises_on_short_wavs_list` becomes `test_generate_batch_short_wavs_list_fails_all_items` (expects per-item `ValueError`s in the result list instead of a raise).

- [ ] **Step 1: Write the failing tests**

In `tests/test_model.py`:

REPLACE `test_generate_batch_raises_on_short_wavs_list` with:
```python
async def test_generate_batch_short_wavs_list_fails_all_items():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    service._model = _ShortResultModel()

    requests = [
        TTSRequest(text="hello", language="English", instruct="calm voice"),
        TTSRequest(text="world", language="English", instruct="calm voice"),
    ]

    results = await service.generate_batch(requests)
    assert all(isinstance(r, ValueError) for r in results)
    assert "wav" in str(results[0])
```

ADD these fakes and tests:
```python
class _FakeCloneModel:
    """Fake Base model: records prompt-build and clone calls."""

    def __init__(self):
        self.prompt_builds: list[str] = []
        self.clone_calls: list[tuple[tuple[str, ...], object]] = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode=False):
        self.prompt_builds.append(ref_text)
        return {"prompt_for": ref_text}

    def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens=None):
        self.clone_calls.append((tuple(text), voice_clone_prompt))
        wavs = [
            np.zeros(max(int(len(t.split()) / 2.5 * 24000), 1), dtype="float32") for t in text
        ]
        return wavs, 24000


class _FakeRegistry:
    def __init__(self, voices: dict):
        self._voices = voices

    def get(self, voice_id):
        return self._voices.get(voice_id)


class _FakeVoiceInfo:
    def __init__(self, ref_text: str):
        self.ref_text = ref_text
        self.wav_path = f"/fake/{ref_text}.wav"


async def test_mixed_batch_splits_design_and_clone_preserving_order():
    settings = Settings(_env_file=None)
    registry = _FakeRegistry({"voice_a": _FakeVoiceInfo("ref a")})
    service = TTSModelService(settings, registry)
    design_model = _FakeModel()
    clone_model = _FakeCloneModel()
    service._model = design_model
    service._clone_model = clone_model

    requests = [
        TTSRequest(text="design one", language="English", instruct="calm voice"),
        TTSRequest(text="clone one", language="English", voice_id="voice_a"),
        TTSRequest(text="design two", language="English", instruct="calm voice"),
        TTSRequest(text="clone two", language="English", voice_id="voice_a"),
    ]

    results = await service.generate_batch(requests)

    assert all(isinstance(r, bytes) for r in results)
    # one clone call for the whole voice_a group, with the cached prompt
    assert clone_model.clone_calls == [
        (("clone one", "clone two"), {"prompt_for": "ref a"})
    ]
    # prompt built exactly once, then cached
    assert clone_model.prompt_builds == ["ref a"]
    await service.generate_batch([requests[1]])
    assert clone_model.prompt_builds == ["ref a"]


async def test_unknown_voice_id_fails_only_its_items():
    settings = Settings(_env_file=None)
    registry = _FakeRegistry({})
    service = TTSModelService(settings, registry)
    service._model = _FakeModel()
    service._clone_model = _FakeCloneModel()

    requests = [
        TTSRequest(text="design item", language="English", instruct="calm voice"),
        TTSRequest(text="clone item", language="English", voice_id="ghost"),
    ]

    results = await service.generate_batch(requests)
    assert isinstance(results[0], bytes)
    assert isinstance(results[1], Exception)
    assert "unknown voice_id" in str(results[1])


async def test_clone_disabled_fails_clone_items_only():
    settings = Settings(_env_file=None, voice_clone_enabled=False)
    service = TTSModelService(settings, _FakeRegistry({}))
    service._model = _FakeModel()

    requests = [
        TTSRequest(text="design item", language="English", instruct="calm voice"),
        TTSRequest(text="clone item", language="English", voice_id="any"),
    ]

    results = await service.generate_batch(requests)
    assert isinstance(results[0], bytes)
    assert isinstance(results[1], Exception)
    assert "voice cloning is disabled" in str(results[1])


async def test_clone_path_goes_through_self_check():
    class _TruncatingCloneModel(_FakeCloneModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate_voice_clone(self, text, language, voice_clone_prompt, max_new_tokens=None):
            self.calls += 1
            if self.calls == 1:
                return [np.zeros(1, dtype="float32") for _ in text], 24000
            return super().generate_voice_clone(text, language, voice_clone_prompt, max_new_tokens)

    settings = Settings(_env_file=None)
    registry = _FakeRegistry({"voice_a": _FakeVoiceInfo("ref a")})
    service = TTSModelService(settings, registry)
    service._model = _FakeModel()
    truncating = _TruncatingCloneModel()
    service._clone_model = truncating

    results = await service.generate_batch(
        [TTSRequest(text="some words here", language="English", voice_id="voice_a")]
    )
    assert isinstance(results[0], bytes)
    assert truncating.calls == 2  # initial + 1 self-check retry
    assert service.self_check_recovered == 1


def test_clone_is_loaded_and_invalidate():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    assert service.clone_is_loaded() is False
    service._clone_model = _FakeCloneModel()
    assert service.clone_is_loaded() is True
    service._clone_prompts["v"] = object()
    service.invalidate_clone_prompt("v")
    assert "v" not in service._clone_prompts
    service.invalidate_clone_prompt("never-there")  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: new tests FAIL (`TypeError` on 2-arg constructor / missing attributes); the replaced short-wavs test FAILS (still raises instead of returning per-item errors).

- [ ] **Step 3: Write the implementation**

In `app/model.py`, replace the `TTSModelService` class body as follows (module-level functions `check_vram`, `_nvidia_smi_processes`, `_wav_to_bytes`, `is_audio_abnormal` are unchanged):

```python
class TTSModelService:
    def __init__(self, settings: Settings, voice_registry=None) -> None:
        self._settings = settings
        self._voices = voice_registry
        self._model = None
        self._clone_model = None
        self._clone_prompts: dict[str, object] = {}
        # Safe without a lock only because BatchWorker's single consumer
        # awaits one _dispatch at a time, so generate_batch never runs
        # concurrently with itself. A second concurrent caller would race
        # these non-atomic increments.
        self.self_check_flagged = 0
        self.self_check_recovered = 0
        self.self_check_exhausted = 0

    def load(self) -> None:
        check_vram(self._settings.device, self._settings.min_free_vram_gb)
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            self._settings.model_id,
            device_map=self._settings.device,
            dtype=torch.bfloat16,
        )
        if self._settings.voice_clone_enabled:
            self._clone_model = Qwen3TTSModel.from_pretrained(
                self._settings.clone_model_id,
                device_map=self._settings.device,
                dtype=torch.bfloat16,
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    def clone_is_loaded(self) -> bool:
        return self._clone_model is not None

    def invalidate_clone_prompt(self, voice_id: str) -> None:
        self._clone_prompts.pop(voice_id, None)

    async def generate_batch(self, requests: Sequence[TTSRequest]) -> list[bytes | Exception]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_batch_sync, list(requests))

    def _generate_batch_sync(self, requests: list[TTSRequest]) -> list[bytes | Exception]:
        results: list[bytes | Exception | None] = [None] * len(requests)

        design_indices = [i for i, r in enumerate(requests) if r.voice_id is None]
        clone_groups: dict[str, list[int]] = {}
        for i, r in enumerate(requests):
            if r.voice_id is not None:
                clone_groups.setdefault(r.voice_id, []).append(i)

        if design_indices:
            self._run_group(requests, design_indices, self._design_generate, results)

        for voice_id, indices in clone_groups.items():
            try:
                prompt = self._get_clone_prompt(voice_id)
            except Exception as exc:  # noqa: BLE001 - fail this group only
                logger.warning("clone prompt unavailable for voice %r: %s", voice_id, exc)
                for i in indices:
                    results[i] = exc
                continue

            def clone_fn(sub_requests, _prompt=prompt):
                return self._clone_model.generate_voice_clone(
                    text=[r.text for r in sub_requests],
                    language=[r.language for r in sub_requests],
                    voice_clone_prompt=_prompt,
                    max_new_tokens=self._settings.max_new_tokens,
                )

            self._run_group(requests, indices, clone_fn, results)

        return results  # type: ignore[return-value]

    def _design_generate(self, sub_requests: list[TTSRequest]):
        return self._model.generate_voice_design(
            text=[r.text for r in sub_requests],
            language=[r.language for r in sub_requests],
            instruct=[r.instruct for r in sub_requests],
            max_new_tokens=self._settings.max_new_tokens,
        )

    def _get_clone_prompt(self, voice_id: str):
        if not self._settings.voice_clone_enabled or self._clone_model is None:
            raise RuntimeError("voice cloning is disabled")
        cached = self._clone_prompts.get(voice_id)
        if cached is not None:
            return cached
        voice = self._voices.get(voice_id) if self._voices is not None else None
        if voice is None:
            raise RuntimeError(f"unknown voice_id: {voice_id}")
        prompt = self._clone_model.create_voice_clone_prompt(
            ref_audio=str(voice.wav_path),
            ref_text=voice.ref_text,
            x_vector_only_mode=False,
        )
        self._clone_prompts[voice_id] = prompt
        return prompt

    def _run_group(
        self,
        requests: list[TTSRequest],
        indices: list[int],
        gen_fn,
        results: "list[bytes | Exception | None]",
    ) -> None:
        """Generate the subset `indices` with gen_fn, run the audio
        self-check + retry loop on it, and write outcomes into `results`
        at the original batch positions. Any failure here affects only
        this group's items."""
        sub = [requests[i] for i in indices]
        try:
            wavs, sr = gen_fn(sub)
            if len(wavs) != len(sub):
                raise ValueError(
                    f"model returned {len(wavs)} wav(s) for {len(sub)} request(s)"
                )
        except Exception as exc:  # noqa: BLE001 - fail this group only
            logger.exception("generation failed for group of %d item(s)", len(sub))
            for i in indices:
                results[i] = exc
            return

        if not self._settings.audio_self_check_enabled:
            for pos, i in enumerate(indices):
                results[i] = _wav_to_bytes(wavs[pos], sr)
            return

        max_wps = self._settings.max_plausible_words_per_second
        last_durations: dict[int, float] = {}
        pending: list[int] = []  # positions within this group
        for pos, i in enumerate(indices):
            wav = wavs[pos]
            last_durations[pos] = len(wav) / sr
            if is_audio_abnormal(wav, sr, requests[i].text, max_wps):
                pending.append(pos)
            else:
                results[i] = _wav_to_bytes(wav, sr)

        if pending:
            self.self_check_flagged += len(pending)
            for pos in pending:
                text = requests[indices[pos]].text
                word_count = len(text.split())
                actual_s = last_durations[pos]
                actual_wps = word_count / actual_s if actual_s > 0 else float("inf")
                logger.info(
                    "audio self-check flagged item: %d words in %.2fs (%.1f words/sec, threshold %.1f)",
                    word_count,
                    actual_s,
                    actual_wps,
                    max_wps,
                )

        for _attempt in range(self._settings.audio_self_check_max_retries):
            if not pending:
                break
            retry_sub = [requests[indices[pos]] for pos in pending]
            try:
                retry_wavs, retry_sr = gen_fn(retry_sub)
                if len(retry_wavs) != len(pending):
                    raise ValueError(
                        f"model returned {len(retry_wavs)} wav(s) "
                        f"for {len(pending)} retry request(s)"
                    )
            except Exception as exc:  # noqa: BLE001 - isolate retry failure to pending items only
                logger.exception("audio self-check retry failed for %d item(s)", len(pending))
                for pos in pending:
                    results[indices[pos]] = exc
                pending = []
                break

            still_pending = []
            for rpos, pos in enumerate(pending):
                wav = retry_wavs[rpos]
                last_durations[pos] = len(wav) / retry_sr
                if is_audio_abnormal(wav, retry_sr, requests[indices[pos]].text, max_wps):
                    still_pending.append(pos)
                else:
                    results[indices[pos]] = _wav_to_bytes(wav, retry_sr)
                    self.self_check_recovered += 1
                    logger.info("audio self-check recovered item on retry %d", _attempt + 1)
            pending = still_pending

        if pending:
            self.self_check_exhausted += len(pending)
            logger.warning(
                "audio self-check exhausted %d retries for %d item(s)",
                self._settings.audio_self_check_max_retries,
                len(pending),
            )
            for pos in pending:
                text = requests[indices[pos]].text
                word_count = len(text.split())
                expected_min_s = word_count / max_wps
                results[indices[pos]] = RuntimeError(
                    f"audio self-check failed after {self._settings.audio_self_check_max_retries} "
                    f"retries: got {last_durations[pos]:.2f}s for {word_count} words "
                    f"(expected >= {expected_min_s:.2f}s)"
                )
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: all pass. Existing self-check tests (`_RecoversOnRetryModel`, `_ControllableModel`, counters, disabled mode, mismatched-length batcher tests) must pass UNCHANGED except the one replaced test — their call-count assertions hold because a single all-design batch is exactly one group.

- [ ] **Step 5: Commit**

```bash
git add app/model.py tests/test_model.py
git commit -m "feat: dual-model batch splitting with clone prompts and group-generic self-check"
```

---

## Task 4: Endpoints, wiring, and API tests

**Files:**
- Modify: `app/main.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_voices_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: module-level `voice_registry` in `app.main`; the three `/v1/voices` endpoints; `/health.clone_model_loaded`; lifespan calls `voice_registry.scan()` before `model_service.load()`.

- [ ] **Step 1: Write the failing tests**

In `tests/conftest.py`, inside the `client` fixture, add (next to the existing `results_dir` patch, BEFORE the `TestClient` entry):
```python
    voices_dir = results_dir.parent / "voices"
    mp.setattr(main_module.settings, "voices_dir", str(voices_dir))
```

Create `tests/test_voices_api.py`:
```python
import io

import numpy as np
import soundfile as sf

from app import main as main_module


def _wav_upload(duration_s: float = 2.0) -> tuple[str, bytes, str]:
    wav = np.zeros(int(duration_s * 24000), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return ("ref.wav", buf.getvalue(), "audio/wav")


def test_register_list_delete_roundtrip(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "test_voice_a", "ref_text": "Hello reference."},
        files={"ref_audio": _wav_upload()},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["voice_id"] == "test_voice_a"
    assert body["duration_s"] == 2.0

    listed = client.get("/v1/voices").json()
    ids = [v["voice_id"] for v in listed["voices"]]
    assert "test_voice_a" in ids

    deleted = client.delete("/v1/voices/test_voice_a")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": "test_voice_a"}

    ids_after = [v["voice_id"] for v in client.get("/v1/voices").json()["voices"]]
    assert "test_voice_a" not in ids_after


def test_register_duplicate_returns_409(client):
    for expected in (201, 409):
        resp = client.post(
            "/v1/voices",
            data={"name": "test_voice_dup", "ref_text": "text"},
            files={"ref_audio": _wav_upload()},
        )
        assert resp.status_code == expected
    client.delete("/v1/voices/test_voice_dup")


def test_register_invalid_audio_returns_422(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "test_voice_bad", "ref_text": "text"},
        files={"ref_audio": ("ref.wav", b"not audio", "audio/wav")},
    )
    assert resp.status_code == 422


def test_register_bad_name_returns_422(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "Bad Name!", "ref_text": "text"},
        files={"ref_audio": _wav_upload()},
    )
    assert resp.status_code == 422


def test_delete_unknown_voice_returns_404(client):
    resp = client.delete("/v1/voices/never_registered")
    assert resp.status_code == 404


def test_health_reports_clone_model(client):
    body = client.get("/health").json()
    assert body["clone_model_loaded"] is False  # load() is a no-op in tests


def test_sync_generation_accepts_voice_id(client):
    # The session fixture's fake _generate_fn bypasses the model service,
    # so this exercises schema + endpoint plumbing for voice_id requests.
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "voice_id": "any_voice"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_sync_generation_rejects_both_instruct_and_voice_id(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "instruct": "calm", "voice_id": "v"},
    )
    assert resp.status_code == 422


def test_job_items_accept_voice_id(client):
    import time

    resp = client.post(
        "/v1/jobs",
        json={"items": [{"text": "hello", "language": "English", "voice_id": "any_voice"}]},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] not in ("pending", "running"):
            break
        time.sleep(0.02)
    assert body["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voices_api.py -v`
Expected: FAIL — 404s for the missing `/v1/voices` routes; `clone_model_loaded` KeyError.

- [ ] **Step 3: Write the implementation**

In `app/main.py`:

Imports — replace the fastapi import line and add the voices import:
```python
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from app import model as model_module
from app.batcher import BatchWorker
from app.config import settings
from app.jobs import ItemStatus, JobManager, job_to_dict
from app.model import TTSModelService
from app.schemas import JobSubmitRequest, TTSRequest
from app.voices import DuplicateVoiceError, InvalidVoiceError, VoiceRegistry
```

Wiring — replace `model_service = TTSModelService(settings)` with:
```python
voice_registry = VoiceRegistry(settings)
model_service = TTSModelService(settings, voice_registry)
```

Lifespan — add the scan as the first startup step:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    voice_registry.scan()
    job_manager.wipe_results_dir()
    model_service.load()
    batch_worker.start()
    yield
    await job_manager.shutdown()
    await batch_worker.stop()
```

`/health` — add the clone field:
```python
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "clone_model_loaded": model_service.clone_is_loaded(),
        "vram_free_gb": vram_free_gb,
        "queue_depth": batch_worker.queue_depth(),
    }
```

New endpoints (after the jobs endpoints, before `if __name__ == "__main__":`):
```python
@app.post("/v1/voices", status_code=201)
async def register_voice(
    name: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
) -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    audio_bytes = await ref_audio.read()
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(
            None, voice_registry.register, name, audio_bytes, ref_text
        )
    except DuplicateVoiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidVoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"voice_id": info.voice_id, "duration_s": round(info.duration_s, 1)}


@app.get("/v1/voices")
async def list_voices() -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    return {
        "voices": [
            {
                "voice_id": v.voice_id,
                "duration_s": round(v.duration_s, 1),
                "ref_text": v.ref_text,
            }
            for v in voice_registry.list_voices()
        ]
    }


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    loop = asyncio.get_running_loop()
    removed = await loop.run_in_executor(None, voice_registry.delete, voice_id)
    if not removed:
        raise HTTPException(status_code=404, detail="voice not found")
    model_service.invalidate_clone_prompt(voice_id)
    return {"deleted": voice_id}
```

- [ ] **Step 4: Run the new tests, then the full suite 2x**

Run: `pytest tests/test_voices_api.py -v` — expect 9 passed.
Run: `for i in 1 2; do pytest -q || break; done` — expect everything passing both times (the pre-existing `test_health_endpoint` must be updated ONLY if it asserts the exact key set — it asserts individual keys, so it should pass unchanged; verify).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/conftest.py tests/test_voices_api.py
git commit -m "feat: add voice registry endpoints and clone wiring"
```

---

## Task 5: Registration script and documentation

**Files:**
- Create: `scripts/register_voices.py`
- Modify: `API.md`
- Modify: `README.md`

No TDD cycle (script is operator tooling verified live by the controller; docs verified by accuracy check).

- [ ] **Step 1: Write the registration script**

Create `scripts/register_voices.py`:
```python
"""Register every {name}.wav + {name}.txt pair in a directory as a voice.

Usage:
    python scripts/register_voices.py /path/to/voice_dir [base_url]
"""

import sys
from pathlib import Path

import httpx


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    voice_dir = Path(sys.argv[1])
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    pairs = sorted(voice_dir.glob("*.wav"))
    if not pairs:
        print(f"no .wav files found in {voice_dir}")
        sys.exit(1)

    ok = failed = 0
    for wav_path in pairs:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"SKIP {wav_path.name}: no matching .txt")
            failed += 1
            continue
        name = wav_path.stem.lower()
        resp = httpx.post(
            f"{base_url}/v1/voices",
            data={"name": name, "ref_text": txt_path.read_text(encoding="utf-8")},
            files={"ref_audio": (wav_path.name, wav_path.read_bytes(), "audio/wav")},
            timeout=60,
        )
        if resp.status_code == 201:
            print(f"OK   {name} ({resp.json()['duration_s']}s)")
            ok += 1
        elif resp.status_code == 409:
            print(f"SKIP {name}: already registered")
            ok += 1
        else:
            print(f"FAIL {name}: {resp.status_code} {resp.text}")
            failed += 1

    print(f"\n{ok} registered/existing, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Verify syntax: `python -m py_compile scripts/register_voices.py` (do NOT run it against a live server — the controller does live verification separately).

- [ ] **Step 2: Update `API.md`**

Add a new section between the Jobs API section and `GET /health` (renumber `GET /health` accordingly), following the file's existing Vietnamese style, documenting:
- `POST /v1/voices` (multipart curl example with `-F name=... -F ref_text=... -F ref_audio=@file.wav`), 201 body, 409/422/503 cases, the slug rule and 0.5–60s duration bounds
- `GET /v1/voices` with response example
- `DELETE /v1/voices/{voice_id}` with 200/404, note that deleting a voice mid-generation may fail in-flight items using it
- In the `POST /v1/tts/voice-design` section: document the new `voice_id` field (exactly-one-of `instruct`/`voice_id`), that `voice_id` requests use the clone model for a stable voice identity, that unknown `voice_id` → 500 (sync) / item `failed` (jobs), and that job items accept `voice_id` identically
- In the "Ghi chú chung": voices persist across restarts (unlike job results); first startup after this feature downloads the ~4GB Base checkpoint

- [ ] **Step 3: Update `README.md`**

- Endpoints list: add the three `/v1/voices` routes with one-line descriptions and a pointer to `API.md`
- Configuration example block: add `QWEN_TTS_CLONE_MODEL_ID`, `QWEN_TTS_VOICES_DIR`, `QWEN_TTS_VOICE_CLONE_ENABLED`
- Operational notes: the server now loads TWO checkpoints (~9-10GB VRAM total); `voices/` is persistent and NOT wiped at startup; first run downloads the Base model (~4GB)

- [ ] **Step 4: Accuracy check**

Re-read both docs against `app/main.py`/`app/voices.py`/`app/config.py` — every route, status code, field name, env var, and bound must match the code exactly. Fix any mismatch.

- [ ] **Step 5: Commit**

```bash
git add scripts/register_voices.py API.md README.md
git commit -m "docs: document voice registry API and add bulk registration script"
```

---

## Post-implementation (controller-run, not a subagent task)

1. Restart the real server (`uvicorn app.main:app --port 8265`) — first start downloads the Base checkpoint (~4GB) and loads both models; verify `/health` shows `model_loaded` and `clone_model_loaded` true and VRAM still has headroom.
2. `python scripts/register_voices.py /home/thanhdev/Downloads/voice_fast http://127.0.0.1:8265` — registers all 24 curated voices.
3. Live clone verification: generate the same sentence twice with one `voice_id`, download both, confirm they decode and (by ear) match the reference voice.

## Self-Review Notes

- **Spec coverage:** settings + request schema → Task 1; persistent registry with scan/skip-malformed → Task 2; dual-model load, lazy prompt cache + invalidation, mixed-batch split, per-group failure isolation, self-check on both paths, disabled-flag behavior → Task 3; endpoints (201/409/422/503/404), `/health.clone_model_loaded`, lifespan scan, executor-offloaded disk I/O, conftest voices_dir patch → Task 4; docs + bulk-registration script → Task 5; live verification + 24-voice registration → controller post-step. All spec sections covered.
- **Placeholder scan:** every code step is complete and runnable; Task 5's doc steps specify content requirements rather than verbatim text (consistent with how the jobs-poll plan handled docs).
- **Type consistency:** `TTSModelService(settings, voice_registry=None)` (Task 3) matches Task 4's `TTSModelService(settings, voice_registry)`; `_get_clone_prompt` uses `VoiceRegistry.get` → `VoiceInfo.wav_path`/`ref_text` exactly as Task 2 defines them; `invalidate_clone_prompt` (Task 3) is called by Task 4's delete endpoint; `clone_is_loaded` (Task 3) feeds `/health` (Task 4). Fakes in Task 3's tests mirror the real `create_voice_clone_prompt`/`generate_voice_clone` signatures verified from package source (recorded in the spec).
- **Known behavior change** (whole-batch initial failure → per-group per-item failure) is explicit in Task 3 with the exact test replacement named. The batcher's own whole-batch path (a raise out of `generate_batch` itself, e.g. a programming error) still exists and its batcher-level tests are unaffected.
