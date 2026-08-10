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
