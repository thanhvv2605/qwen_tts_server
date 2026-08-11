# Voice Registry + Clone Generation — Design

Date: 2026-08-11
Status: Approved

## Purpose

The VoiceDesign model generates a slightly different voice on every call
(sampling; no seed support — verified in the `qwen_tts` package source).
The client needs MANY audios in the SAME voice (e.g. 578 shorts per
channel, one consistent narrator per topic/language).

The path the model family actually supports for this is **voice cloning**:
the `Qwen3-TTS-12Hz-1.7B-Base` model clones a voice from a short reference
WAV + its transcript, and a `voice_clone_prompt` built once from that
reference can be reused across unlimited generate calls with a stable
voice identity.

This feature adds a **persistent voice registry** (register a reference
audio once → get a `voice_id`) and **clone-based generation** (any
existing generation endpoint accepts `voice_id` instead of `instruct`),
so every audio generated with a given `voice_id` comes out in the same
voice. The user has already produced 24 curated reference voices
(8 personas x EN/DE/FR) with matching transcripts, ready to register.

## Verified package facts this design relies on

(From direct source inspection of the installed `qwen_tts` package.)

- `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", ...)`
  loads the clone-capable model; clone methods raise unless
  `tts_model_type == "base"`, so this is a SEPARATE checkpoint from
  VoiceDesign — both must be resident in VRAM (~4-5GB each; the RTX 3090
  hosts both comfortably).
- `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)`
  → `list[VoiceClonePromptItem]`. Built once per voice, reusable across
  calls. `x_vector_only_mode=False` (ICL mode, needs `ref_text`) gives the
  best quality — we always have `ref_text`, so always use ICL mode.
- `generate_voice_clone(text=[...], language=[...], voice_clone_prompt=items)`
  → `(wavs, sample_rate)`. Batch of multiple texts is supported with ONE
  prompt (one voice) per call — so mixed-voice batches must be grouped by
  voice and cloned one group per call.
- `generate_voice_clone` has NO `instruct` parameter — voice-clone requests
  carry `text` + `language` + `voice_id` only.

## API changes

### New: voice registry

`POST /v1/voices` — multipart form upload:
- `name` (form field): the voice id, client-chosen slug, `[a-z0-9_-]{1,64}`
  (e.g. `astronomy_male_en`). `409` if it already exists.
- `ref_audio` (file field): WAV file (validated by decoding with
  `soundfile`; must be 0.5s–60s long → `422` otherwise).
- `ref_text` (form field): the exact transcript of the reference audio,
  non-empty → `422` otherwise.

Response `201`:
```json
{"voice_id": "astronomy_male_en", "duration_s": 11.8}
```

`GET /v1/voices` → `200`:
```json
{"voices": [{"voice_id": "astronomy_male_en", "duration_s": 11.8, "ref_text": "Far beyond..."}]}
```

`DELETE /v1/voices/{voice_id}` → `200` (removed) / `404` (unknown).
Deleting a voice that an in-flight generation is using is allowed; that
generation may fail with a per-item error (acceptable race, documented).

### Changed: `TTSRequest` (affects sync endpoint AND job items)

`voice_id: str | None = None` is added. Validation becomes:
- **Exactly one** of `instruct` / `voice_id` must be provided.
  - `instruct` set → voice-design generation (existing behavior, unchanged).
  - `voice_id` set → clone generation with the registered voice.
  - both set or neither set → `422`.
- `text`/`language` rules unchanged.
- A `voice_id` that doesn't exist in the registry is NOT rejected at
  request validation (the registry lives server-side and can change);
  it fails at generation time with a per-item error → sync endpoint `500`,
  job item `failed` with a clear "unknown voice_id" message.

The Jobs API inherits `voice_id` support automatically (job items ARE
`TTSRequest`s) — this is the primary intended usage: one job of 578 items
all carrying the same `voice_id`.

## Storage

```
VOICES_DIR/                      (default ./voices, env QWEN_TTS_VOICES_DIR)
  astronomy_male_en/
    ref.wav
    ref.txt
```

- **Persistent** — NOT wiped at startup (unlike `RESULTS_DIR`). Voices
  survive restarts; the registry is rebuilt by scanning `VOICES_DIR` at
  startup.
- Clone prompts (`VoiceClonePromptItem` tensors) are NOT persisted — they
  are built lazily on each voice's first use after startup and cached in
  memory. Prompt building runs inside the generation executor thread
  (it does feature extraction — not free, but one-time per voice per
  process lifetime).
- `voices/` added to `.gitignore`.

## Model loading

`TTSModelService.load()` now loads BOTH checkpoints:
1. VoiceDesign (`settings.model_id`, unchanged)
2. Base (`settings.clone_model_id`, default
   `Qwen/Qwen3-TTS-12Hz-1.7B-Base`) — only when
   `settings.voice_clone_enabled` (default `true`).

When `voice_clone_enabled=false`: Base is not loaded, `/v1/voices`
endpoints return `503 {"detail": "voice cloning is disabled"}`, and any
generation request carrying `voice_id` fails per-item with the same
message. (Kill-switch pattern, mirrors the self-check flag.)

`/health` gains `clone_model_loaded: bool`.

## Generation flow (mixed batches)

`TTSModelService._generate_batch_sync` currently sends the whole batch to
`generate_voice_design`. It becomes a splitter:

```
batch (up to MAX_BATCH_SIZE items, mixed)
  ├─ design items (instruct set)  → one generate_voice_design call (as today)
  └─ clone items (voice_id set)   → grouped by voice_id
       → one generate_voice_clone call per group, using the cached
         (lazily-built) clone prompt for that voice
results reassembled in original batch order → list[bytes | Exception]
```

- An unknown/deleted `voice_id`, a failed prompt build, or a failed clone
  call produces per-item `Exception`s for that group only — sibling items
  (other groups, design items) are unaffected. This composes with the
  existing per-item failure isolation in `BatchWorker`.
- The **audio self-check + regenerate** applies to clone output exactly as
  it does to design output (same word-count heuristic, same retry loop,
  same per-item failure after exhaustion) — retries regenerate only the
  abnormal subset within each path.
- `BatchWorker` itself is UNCHANGED (same `GenerateFn` contract).

## New settings (`app/config.py`, env prefix `QWEN_TTS_`)

| Field | Type | Default |
|---|---|---|
| `clone_model_id` | `str` | `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"` |
| `voices_dir` | `str` | `"./voices"` |
| `voice_clone_enabled` | `bool` | `true` |

## Error handling

- Registration: invalid slug / undecodable audio / out-of-range duration /
  empty ref_text → `422`; duplicate name → `409`; registry disabled → `503`.
- Generation with unknown `voice_id` → per-item
  `RuntimeError("unknown voice_id: <id>")` → sync `500` / job item `failed`.
- Generation with `voice_id` while cloning disabled → per-item
  `RuntimeError("voice cloning is disabled")`.
- Prompt build failure (corrupt ref file etc.) → per-item error for that
  group; the voice's cache entry is not poisoned (next use retries the
  build).
- Startup scan skips (with a warning log) any `VOICES_DIR` entry missing
  `ref.wav`/`ref.txt` rather than failing boot.

## Testing

- Schema: exactly-one-of `instruct`/`voice_id` validation matrix (both /
  neither / each alone).
- VoiceRegistry unit tests (no GPU): register/list/delete round-trip on a
  tmp dir; duplicate name; invalid audio bytes; duration bounds; startup
  rescan finds previously registered voices; corrupt-entry skip.
- Model service split logic with fake models (no GPU): mixed batch
  reassembles results in original order; per-group failure isolation
  (unknown voice_id fails only its items); clone path goes through
  self-check retries; disabled flag fails clone items but not design items.
- Endpoint tests on the existing session `TestClient`: register via
  multipart → list → generate (sync + a job item) with `voice_id` using a
  fake clone path → delete → generate again fails per-item.
- Live verification after deployment (controller-run, real GPU): register
  one real voice from `/home/thanhdev/Downloads/voice_fast`, generate the
  same sentence twice with the `voice_id`, confirm both outputs decode and
  the flow works end-to-end; then register all 24 voices via script.

## Out of scope

- Numeric speed control (post-process with ffmpeg atempo if needed).
- Word/sentence timestamps (next feature per agreed priority order).
- Editing a registered voice in place (delete + re-register instead).
- Persisting built clone prompts to disk (rebuild-on-first-use is fine).
- CustomVoice model / its 9 built-in speakers.
- Authentication (unchanged: internal server).
