# Voice-Design Kill Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `QWEN_TTS_VOICE_DESIGN_ENABLED=false` runs the server clone-only
(skips the VoiceDesign checkpoint, reclaiming ~4-5GB VRAM); `instruct`
requests then fail per-item with a clear message while `voice_id` requests
work unchanged.

**Architecture:** Exact mirror of the existing `voice_clone_enabled` kill
switch: a Settings flag, a load-time skip, a per-item guard in
`_generate_batch_sync`'s design path, an adjusted VRAM threshold (+6 only
when BOTH models are enabled), and a widened `vram_free_gb` gate in
`/health`. Design load stays fail-fast when enabled.

**Tech Stack:** unchanged.

**Spec:** `docs/superpowers/specs/2026-08-11-voice-design-kill-switch-design.md`

## Global Constraints

- `voice_design_enabled: bool = True` (env `QWEN_TTS_VOICE_DESIGN_ENABLED`)
- Disabled design → design items fail per-item with `RuntimeError("voice design is disabled")`; clone items in the same batch unaffected
- `load()`: design checkpoint loaded only when enabled (fail-fast when enabled, unchanged); VRAM threshold `min_free + 6.0` ONLY when both models enabled, else `min_free`; neither enabled → skip check and load nothing
- `/health`: `model_loaded` semantics unchanged; `vram_free_gb` computed when `is_loaded() or clone_is_loaded()`
- Only `app/config.py`, `app/model.py`, `app/main.py`, their test files, and docs (`README.md`, `API.md`) may change

---

## Task 1: Kill switch (single task — settings, model service, health, tests, docs)

**Files:**
- Modify: `app/config.py`, `app/model.py`, `app/main.py`
- Modify: `tests/test_config.py`, `tests/test_model.py`
- Modify: `README.md`, `API.md`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py` — extend `test_default_settings` with:
```python
    assert settings.voice_design_enabled is True
```
and add:
```python
def test_env_override_voice_design_enabled(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_VOICE_DESIGN_ENABLED", "false")
    settings = Settings(_env_file=None)
    assert settings.voice_design_enabled is False
```

`tests/test_model.py` — add:
```python
async def test_design_disabled_fails_design_items_only():
    settings = Settings(_env_file=None, voice_design_enabled=False)
    registry = _FakeRegistry({"voice_a": _FakeVoiceInfo("ref a")})
    service = TTSModelService(settings, registry)
    service._clone_model = _FakeCloneModel()

    requests = [
        TTSRequest(text="design item", language="English", instruct="calm voice"),
        TTSRequest(text="clone item", language="English", voice_id="voice_a"),
    ]

    results = await service.generate_batch(requests)
    assert isinstance(results[0], Exception)
    assert "voice design is disabled" in str(results[0])
    assert isinstance(results[1], bytes)


def test_load_selects_models_by_flags(monkeypatch):
    import sys
    import types

    loaded: list[str] = []

    class _FakeQwenModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loaded.append(model_id)
            return object()

    fake_module = types.SimpleNamespace(Qwen3TTSModel=_FakeQwenModel)
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_module)
    monkeypatch.setattr("app.model.check_vram", lambda device, min_free_gb: 20.0)

    settings = Settings(_env_file=None, voice_design_enabled=False)
    service = TTSModelService(settings)
    service.load()
    assert loaded == [settings.clone_model_id]
    assert service.is_loaded() is False
    assert service.clone_is_loaded() is True

    loaded.clear()
    checked: list[float] = []
    monkeypatch.setattr(
        "app.model.check_vram",
        lambda device, min_free_gb: checked.append(min_free_gb),
    )
    both_off = Settings(
        _env_file=None, voice_design_enabled=False, voice_clone_enabled=False
    )
    TTSModelService(both_off).load()
    assert loaded == []
    assert checked == []  # neither model enabled -> VRAM check skipped
```

Update the existing threshold test `test_load_vram_threshold_scales_with_clone_enabled` to cover the new matrix (replace its body):
```python
def test_load_vram_threshold_scales_with_enabled_models(monkeypatch):
    checked = []

    def fake_check_vram(device, min_free_gb):
        checked.append(min_free_gb)
        raise RuntimeError("stop before model load")

    monkeypatch.setattr("app.model.check_vram", fake_check_vram)

    with pytest.raises(RuntimeError):
        TTSModelService(Settings(_env_file=None)).load()  # both enabled
    with pytest.raises(RuntimeError):
        TTSModelService(Settings(_env_file=None, voice_clone_enabled=False)).load()
    with pytest.raises(RuntimeError):
        TTSModelService(Settings(_env_file=None, voice_design_enabled=False)).load()

    assert checked == [12.0, 6.0, 6.0]
```
(Rename the old test to this name; keep exactly one threshold test.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py tests/test_model.py -v`
Expected: FAIL — `AttributeError: voice_design_enabled`; new model tests fail on missing flag behavior.

- [ ] **Step 3: Implement**

`app/config.py` — add after `voice_clone_enabled`:
```python
    voice_clone_enabled: bool = True
    voice_design_enabled: bool = True
```

`app/model.py` — replace `load()` with:
```python
    def load(self) -> None:
        design_on = self._settings.voice_design_enabled
        clone_on = self._settings.voice_clone_enabled
        if not design_on and not clone_on:
            logger.warning("both voice design and voice cloning are disabled; loading no models")
            return
        required_gb = self._settings.min_free_vram_gb
        if design_on and clone_on:
            # Two 1.7B bf16 checkpoints + activations peak around 11-12GB;
            # the single-model threshold alone would warn far too late.
            required_gb += 6.0
        check_vram(self._settings.device, required_gb)
        from qwen_tts import Qwen3TTSModel

        if design_on:
            self._model = Qwen3TTSModel.from_pretrained(
                self._settings.model_id,
                device_map=self._settings.device,
                dtype=torch.bfloat16,
            )
        if clone_on:
            try:
                self._clone_model = Qwen3TTSModel.from_pretrained(
                    self._settings.clone_model_id,
                    device_map=self._settings.device,
                    dtype=torch.bfloat16,
                )
            except Exception:  # noqa: BLE001 - degrade to clone-unavailable, keep design serving
                logger.exception(
                    "failed to load clone model %r; voice cloning will be unavailable",
                    self._settings.clone_model_id,
                )
                self._clone_model = None
```
(Check the current `load()` first — preserve the existing clone try/except exactly; the design load body is the existing one moved under `if design_on`.)

`app/model.py` — in `_generate_batch_sync`, replace the design-group dispatch:
```python
        if design_indices:
            if not self._settings.voice_design_enabled or self._model is None:
                exc: Exception = RuntimeError("voice design is disabled")
                for i in design_indices:
                    results[i] = exc
            else:
                self._run_group(requests, design_indices, self._design_generate, results)
```

`app/main.py` — in `health()`, widen the gate:
```python
    if model_service.is_loaded() or model_service.clone_is_loaded():
```
(everything else in the endpoint unchanged).

Docs:
- `README.md`: add `QWEN_TTS_VOICE_DESIGN_ENABLED` to the config example
  (comment: tắt để chạy chỉ-clone, tiết kiệm ~4-5GB VRAM); operational note
  that clone-only mode loads one checkpoint and `scripts/smoke_test.py`
  (design path) is not applicable in that mode.
- `API.md`: general notes — mention the flag; `500` section — add
  "voice design is disabled" as a cause when the flag is off and an
  `instruct` request arrives.

- [ ] **Step 4: Run the full suite 2x**

Run: `for i in 1 2; do pytest -q || break; done`
Expected: all passing both runs (~106 tests).

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/model.py app/main.py tests/test_config.py tests/test_model.py README.md API.md
git commit -m "feat: add voice-design kill switch for clone-only serving"
```

---

## Post-implementation (controller-run)

Restart the real server with `QWEN_TTS_VOICE_DESIGN_ENABLED=false`; verify:
VRAM usage drops ~4-5GB vs dual-model; `/health` shows `model_loaded: false`,
`clone_model_loaded: true`, `vram_free_gb` populated; clone gen with a
registered voice_id → 200; `instruct` request → `500 "voice design is disabled"`.

## Self-Review Notes

- Spec coverage: flag → Step 3 config; load-skip + threshold matrix + neither-enabled skip → Step 3 `load()` + both new tests; per-item design guard → Step 3 model + mirror test; health gate → Step 3 main; docs → Step 3 docs. All covered.
- Type consistency: guard uses the same per-item `RuntimeError` value pattern `_generate_batch_sync` already uses for clone groups; no interface changes for batcher/jobs.
- The existing `test_load_vram_threshold_scales_with_clone_enabled` is renamed/extended rather than duplicated — exactly one threshold test remains.
