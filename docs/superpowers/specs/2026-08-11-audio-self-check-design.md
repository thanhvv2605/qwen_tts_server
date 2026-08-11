# Audio Self-Check & Regenerate — Design

Date: 2026-08-11
Status: Approved

## Purpose

Production usage of the VoiceDesign TTS server surfaced a real failure mode:
even for officially-supported languages (English), roughly **20% of
requests** return audio that is implausibly short for the input text (e.g.
a 35-word sentence returning 0.33s of audio — a truncated/degenerate
generation, not a slow-speech artifact).

Today the client app detects this itself (by comparing audio duration to
expected speech length) and re-requests over the network. That costs one
wasted GPU generation **and** one wasted network round-trip per failure.
This feature moves detection + regeneration onto the server, where a
regenerate is just another local GPU call — no network round-trip, and
the check runs on data the server already has in hand before it ever
replies to the client.

## Problem context (this codebase)

- `app/model.py`'s `TTSModelService._generate_batch_sync` calls
  `self._model.generate_voice_design(text=[...], language=[...],
  instruct=[...], ...)` once per batch and returns one WAV per input, with
  **no built-in correctness check** — whatever the model returns is
  returned to the client as-is.
- `app/batcher.py`'s `BatchWorker` batches up to `MAX_BATCH_SIZE` (default
  4) requests together into one `generate_fn` call. Today, `generate_fn`'s
  contract is **all-or-nothing**: it returns `list[bytes]` (all succeeded)
  or raises (the whole batch's requests all fail with the same error).
  This is enforced in `_dispatch` (`app/batcher.py:84-100`).

## Requirement: per-item failure isolation

If one request in a batch is unrecoverably bad (still abnormal after
retries), the other requests in the **same batch** must still succeed
normally — they must not be forced to fail and retry over the network
just because an unrelated sibling in their batch had a bad generation.
(Decided explicitly: this is the whole point of moving the check
server-side — don't reintroduce round-trips by another path.)

This requires extending `BatchWorker`'s `generate_fn` contract from
all-or-nothing to per-item: a batch call can now return a **mix** of
successful bytes and per-item failures.

## Design

### 1. Detection heuristic

```
word_count = len(text.split())
expected_min_duration_s = word_count / MAX_PLAUSIBLE_WORDS_PER_SECOND
actual_duration_s = len(wav) / sample_rate
abnormal = actual_duration_s < expected_min_duration_s
```

`MAX_PLAUSIBLE_WORDS_PER_SECOND` defaults to `4.5` — generous enough to
never false-positive on genuinely fast natural speech, but low enough to
catch truncation (the reported case: 35 words / 0.33s ≈ 106 words/sec,
~24x over the threshold).

This is a pure function of `(wav, sample_rate, text)` — no GPU, no I/O —
so it's directly unit-testable.

### 2. Regenerate flow

Runs inside `TTSModelService._generate_batch_sync` (already executed in a
threadpool executor via `run_in_executor` — see `app/model.py:72-74` — so
retries add latency to that batch only, never block the event loop):

```
1. Generate the whole batch once (existing behavior).
2. For each item, run the detection heuristic.
3. If any items are abnormal: regenerate ONLY those items (a fresh
   generate_voice_design call with just the abnormal subset's
   text/language/instruct) — not the whole batch.
4. Repeat step 3 up to AUDIO_SELF_CHECK_MAX_RETRIES times (default 2)
   total regenerate attempts.
5. Any item still abnormal after all retries becomes a per-item failure
   (see below) — it does not affect sibling items in the batch.
```

Regenerating relies on the model's normal sampling (`do_sample`, the
default) to plausibly produce a different, hopefully-complete result on
retry — no new model parameters are needed for this.

A retry call that itself fails (raises, or returns a wav list shorter
than the pending subset it was asked to regenerate) fails only the
still-pending items it was called for — siblings whose good results are
already recorded are unaffected.

### 3. `BatchWorker` contract change (`app/batcher.py`)

`GenerateFn`'s return type changes from `list[bytes]` to
`list[bytes | Exception]`:

```python
GenerateFn = Callable[[Sequence[TTSRequest]], Awaitable[list["bytes | Exception"]]]
```

`_dispatch` (`app/batcher.py:84-100`) changes its per-item distribution
loop:

```python
for i, result in zip(batch, results):
    if i.future.done():
        continue
    if isinstance(result, BaseException):
        i.future.set_exception(result)
    else:
        i.future.set_result(result)
```

The **existing** whole-batch failure path (the `try/except Exception`
around `await self._generate_fn(requests)`) is unchanged and still
handles the case where `generate_fn` itself raises (e.g. the GPU call
crashes entirely, not a per-item quality issue) — that still fails every
request in the batch, as today. This is additive, not a replacement: two
failure paths now exist for two different failure classes.

`_generate_batch_sync` returns a per-item `Exception` instance (not a
raised exception) for any item still abnormal after retries:

```python
RuntimeError(
    f"audio self-check failed after {settings.audio_self_check_max_retries} "
    f"retries: got {actual_duration_s:.2f}s for {word_count} words "
    f"(expected >= {expected_min_duration_s:.2f}s)"
)
```

This propagates to the client as today's existing `500 {"detail": "..."}`
shape (`app/main.py:42-44` already catches any exception from
`batch_worker.submit()` and maps it to 500) — no change needed in
`app/main.py`.

### 4. New settings (`app/config.py`, env prefix `QWEN_TTS_` as today)

| Field | Type | Default |
|---|---|---|
| `max_plausible_words_per_second` | `float` | `4.5` |
| `audio_self_check_max_retries` | `int` | `2` |
| `audio_self_check_enabled` | `bool` | `true` |

## Data flow (updated)

```
BatchWorker collects batch → TTSModelService._generate_batch_sync:
  generate whole batch once
  → detect abnormal items (pure function, no GPU)
  → loop: regenerate only abnormal subset (up to N retries)
  → build list[bytes | Exception], one per original item, same order
→ BatchWorker._dispatch distributes: bytes → future.set_result,
  Exception → future.set_exception (per item, independent of siblings)
```

## Testing

- **Unit test for the detection heuristic** (pure function, no GPU): known
  short/abnormal case (35 words, 0.33s) → `True`; a plausible duration for
  the same word count → `False`; boundary case at exactly the threshold.
- **Unit test for `TTSModelService._generate_batch_sync`'s retry loop**:
  inject a fake model whose `generate_voice_design` returns a bad (short)
  wav for a given text on its first N calls and a good one after —
  confirm the final returned list has the corrected result for that item,
  and confirm `generate_voice_design` was called the expected number of
  times (whole batch once, then only the abnormal subset on each retry —
  not the whole batch again).
  Also inject a model that **never** produces a good result for one item
  — confirm that item's slot is an `Exception` while sibling items (which
  succeeded on generation 1) are still `bytes`.
- **Unit test for `BatchWorker._dispatch`'s updated distribution logic**:
  `generate_fn` returns a mixed list (`[bytes, Exception(...), bytes]`) —
  confirm each future resolves independently (2 succeed, 1 raises),
  reusing the existing `BatchWorker` test harness/fakes from
  `tests/test_batcher.py`. Also confirm the pre-existing whole-batch
  failure path (raising *out of* `generate_fn` entirely) still fails
  every item, unchanged.
- Given `app/batcher.py` is concurrency-sensitive code that was already
  hardened once by a high-scrutiny review pass, this change goes through
  the same level of review rigor (most-capable reviewer model, explicit
  focus on races/cancellation) before being considered done.

## Out of scope

- Regeneration strategy tuning (e.g. bumping `temperature` on retry) —
  retries reuse the same generation parameters as the original call.
- An upper-duration check (implausibly *long* audio / rambling) — not the
  reported failure mode; not built unless it becomes one.
- Any change to the `POST /v1/tts/voice-design` request/response shape —
  this is entirely a server-internal reliability improvement.
- The other two production requests from the same feedback (fixed voice
  identity + numeric speed control; sentence-level timestamps) — each is
  its own separate design/plan cycle per the agreed priority order
  (self-check → fixed voice/speed → timestamps).
- Language-specific accuracy of the word-count heuristic. It counts
  whitespace-separated tokens, so for Vietnamese it effectively counts
  syllables rather than words (typical measured rate is ~3.1 syllables/sec
  against the 4.5 threshold — only ~1.4x headroom, thinner than the
  English case this feature was designed around; watch the new
  `self-check flagged` logs before trusting the default in production).
  It is effectively inert for Chinese/Japanese/Korean, which have no
  whitespace between words, so `word_count` collapses to ~1 regardless of
  actual sentence length.
- Per-language threshold recalibration — deferred pending production log
  data from the new observability counters/log lines.
