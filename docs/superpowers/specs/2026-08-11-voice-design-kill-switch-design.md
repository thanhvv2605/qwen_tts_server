# Voice-Design Kill Switch — Design

Date: 2026-08-11
Status: Approved

## Purpose

The user's production workflow now runs entirely on **fixed voices**
(`voice_id` / clone generation). The VoiceDesign checkpoint still loads at
startup and holds ~4-5GB of VRAM it never uses in this mode. This feature
adds a kill switch — mirroring the existing `voice_clone_enabled` — so the
server can run clone-only and reclaim that VRAM. Voice design can be
re-enabled any time a new reference voice needs to be created.

## Design

### New setting (`app/config.py`, env prefix `QWEN_TTS_`)

| Field | Type | Default |
|---|---|---|
| `voice_design_enabled` | `bool` | `true` |

Default `true` — existing deployments are unaffected.

### `TTSModelService.load()` (`app/model.py`)

- Loads the VoiceDesign checkpoint only when `voice_design_enabled`.
- The design load stays **fail-fast** when enabled (unchanged original
  contract); only the clone load has the degrade-to-unavailable wrapper.
- VRAM threshold: `min_free_vram_gb` covers ONE model; the `+6.0` bump
  applies only when BOTH models are enabled:
  `required = min_free + (6.0 if design_enabled and clone_enabled else 0)`.
- If NEITHER model is enabled, `load()` skips the VRAM check and loads
  nothing (degenerate config; `/health` shows both flags false).

### Generation (`app/model.py`)

In `_generate_batch_sync`, before running the design group:
```python
if design_indices:
    if not self._settings.voice_design_enabled or self._model is None:
        exc = RuntimeError("voice design is disabled")
        for i in design_indices:
            results[i] = exc
    else:
        self._run_group(requests, design_indices, self._design_generate, results)
```
Per-item failure, clone groups unaffected — the exact mirror of how clone
items fail when cloning is disabled. Surfaces as `500 {"detail": "voice
design is disabled"}` on the sync endpoint and item `failed` in jobs.

### `/health` (`app/main.py`)

- `model_loaded` keeps its meaning (design model loaded) — honest `false`
  when design is disabled.
- `vram_free_gb` gate widens from `is_loaded()` to
  `is_loaded() or clone_is_loaded()` so clone-only mode still reports VRAM.

### Docs

- README: config var + operational note (clone-only mode ≈ one checkpoint
  ≈ 4-5GB VRAM; `scripts/smoke_test.py` targets the design path and is not
  applicable in clone-only mode).
- API.md: general notes mention the flag; `500` section mentions
  "voice design is disabled" as a cause when the flag is off.

## Testing

- Config: default + env override.
- Model service: design-disabled batch → design items fail per-item with
  "voice design is disabled", clone items in the same batch succeed
  (mirror of `test_clone_disabled_fails_clone_items_only`).
- `load()` model selection: with a fake `qwen_tts` module injected into
  `sys.modules` recording `from_pretrained` calls (and `check_vram`
  no-oped), assert design-disabled loads only `clone_model_id`, and
  both-disabled loads nothing and skips the VRAM check.
- VRAM threshold: extend the existing threshold test —
  both enabled → `min_free + 6`; clone-only → `min_free`;
  design-only → `min_free`.
- Live verification (controller): restart the real server with
  `QWEN_TTS_VOICE_DESIGN_ENABLED=false`, confirm VRAM drops ~4-5GB, the 24
  registered voices still work (clone gen 200), and an `instruct` request
  returns `500 "voice design is disabled"`.

## Out of scope

- Degrading a FAILED design load to unavailable (design stays fail-fast).
- Changing `scripts/smoke_test.py` (documented as design-mode-only).
- Renaming `/health.model_loaded` (breaking change for clients).
