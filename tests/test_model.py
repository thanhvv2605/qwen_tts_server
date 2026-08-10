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
