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
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
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

    def is_loaded(self) -> bool:
        return self._model is not None

    async def generate_batch(self, requests: Sequence[TTSRequest]) -> list[bytes | Exception]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_batch_sync, list(requests))

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

        if len(wavs) != len(requests):
            raise ValueError(
                f"generate_voice_design returned {len(wavs)} wav(s) for {len(requests)} request(s)"
            )

        if not self._settings.audio_self_check_enabled:
            return [_wav_to_bytes(wav, sr) for wav in wavs]

        max_wps = self._settings.max_plausible_words_per_second
        results: list[bytes | Exception | None] = [None] * len(requests)
        last_durations: dict[int, float] = {}
        pending: list[int] = []
        for i, wav in enumerate(wavs):
            last_durations[i] = len(wav) / sr
            if is_audio_abnormal(wav, sr, texts[i], max_wps):
                pending.append(i)
            else:
                results[i] = _wav_to_bytes(wav, sr)

        if pending:
            self.self_check_flagged += len(pending)
            for i in pending:
                word_count = len(texts[i].split())
                actual_s = last_durations[i]
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
            try:
                retry_wavs, retry_sr = self._model.generate_voice_design(
                    text=[texts[i] for i in pending],
                    language=[languages[i] for i in pending],
                    instruct=[instructs[i] for i in pending],
                    max_new_tokens=self._settings.max_new_tokens,
                )
                if len(retry_wavs) != len(pending):
                    raise ValueError(
                        f"generate_voice_design returned {len(retry_wavs)} wav(s) "
                        f"for {len(pending)} retry request(s)"
                    )
            except Exception as exc:  # noqa: BLE001 - isolate retry failure to pending items only
                logger.exception("audio self-check retry failed for %d item(s)", len(pending))
                for i in pending:
                    results[i] = exc
                pending = []
                break

            still_pending = []
            for pos, i in enumerate(pending):
                wav = retry_wavs[pos]
                last_durations[i] = len(wav) / retry_sr
                if is_audio_abnormal(wav, retry_sr, texts[i], max_wps):
                    still_pending.append(i)
                else:
                    results[i] = _wav_to_bytes(wav, retry_sr)
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
            for i in pending:
                word_count = len(texts[i].split())
                expected_min_s = word_count / max_wps
                results[i] = RuntimeError(
                    f"audio self-check failed after {self._settings.audio_self_check_max_retries} "
                    f"retries: got {last_durations[i]:.2f}s for {word_count} words "
                    f"(expected >= {expected_min_s:.2f}s)"
                )

        return results  # type: ignore[return-value]
