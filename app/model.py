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


def is_audio_abnormal(
    wav: np.ndarray, sample_rate: int, text: str, max_plausible_words_per_second: float
) -> bool:
    word_count = len(text.split())
    expected_min_duration_s = word_count / max_plausible_words_per_second
    actual_duration_s = len(wav) / sample_rate
    return actual_duration_s < expected_min_duration_s


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
            if not self._settings.voice_design_enabled or self._model is None:
                exc: Exception = RuntimeError("voice design is disabled")
                for i in design_indices:
                    results[i] = exc
            else:
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
        if not self._settings.voice_clone_enabled:
            raise RuntimeError("voice cloning is disabled")
        if self._clone_model is None:
            raise RuntimeError("voice cloning is unavailable: clone model failed to load")
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
