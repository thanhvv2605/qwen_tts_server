import io
import logging

import numpy as np
import pytest
import soundfile as sf
import torch

from app.config import Settings
from app.model import TTSModelService, _wav_to_bytes, check_vram, is_audio_abnormal
from app.schemas import TTSRequest


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
        # Duration well above the self-check threshold (2.5 words/sec is
        # slower than the 4.5 words/sec threshold, so this never trips
        # the self-check regardless of input text length).
        wavs = [
            np.zeros(max(int(len(t.split()) / 2.5 * 24000), 1), dtype="float32") for t in text
        ]
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


async def test_generate_batch_regenerates_abnormal_item_and_keeps_siblings(caplog):
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    model = _RecoversOnRetryModel()
    service._model = model

    requests = [
        TTSRequest(text="good one", language="English", instruct="calm voice"),
        TTSRequest(text="flaky", language="English", instruct="calm voice"),
    ]

    with caplog.at_level(logging.INFO):
        results = await service.generate_batch(requests)

    assert len(results) == 2
    assert isinstance(results[0], bytes)
    assert isinstance(results[1], bytes)
    # First call: whole batch. Second call: only the still-abnormal subset ("flaky").
    assert model.call_texts == [("good one", "flaky"), ("flaky",)]

    assert service.self_check_flagged == 1
    assert service.self_check_recovered == 1
    assert service.self_check_exhausted == 0
    assert "self-check" in caplog.text


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

    assert service.self_check_flagged == 1
    assert service.self_check_recovered == 0
    assert service.self_check_exhausted == 1
    # Spec-compliant message: duration/word-count diagnostics only, no leaked user text.
    assert "always broken" not in str(results[1])


class _ShortResultModel:
    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        # Returns one fewer wav than requested - a model contract violation.
        wavs = [np.zeros(24000, dtype="float32") for _ in text[:-1]]
        return wavs, 24000


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


async def test_generate_batch_skips_self_check_when_disabled():
    settings = Settings(_env_file=None, audio_self_check_enabled=False)
    service = TTSModelService(settings)
    model = _ControllableModel(always_bad_texts=frozenset({"always broken"}))
    service._model = model

    requests = [
        TTSRequest(text="good one", language="English", instruct="calm voice"),
        TTSRequest(text="always broken", language="English", instruct="calm voice"),
    ]

    results = await service.generate_batch(requests)

    assert len(results) == 2
    assert isinstance(results[0], bytes)
    assert isinstance(results[1], bytes)
    # No retries: exactly one call to the model, no self-check at all.
    assert len(model.call_texts) == 1
    assert service.self_check_flagged == 0
    assert service.self_check_recovered == 0
    assert service.self_check_exhausted == 0


class _RetryRaisesModel:
    """First call: one good item, one truncated item. Second call (the
    retry, for the truncated item only) raises."""

    def __init__(self):
        self.call_texts: list[tuple[str, ...]] = []

    def generate_voice_design(self, text, language, instruct, max_new_tokens=None):
        self.call_texts.append(tuple(text))
        if len(self.call_texts) == 1:
            wavs = []
            for t in text:
                if t == "flaky":
                    wavs.append(np.zeros(1, dtype="float32"))
                else:
                    word_count = len(t.split())
                    wavs.append(np.zeros(max(int(word_count / 2.5 * 24000), 1), dtype="float32"))
            return wavs, 24000
        raise RuntimeError("gpu hiccup")


async def test_generate_batch_isolates_retry_call_failure_to_pending_items():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    model = _RetryRaisesModel()
    service._model = model

    requests = [
        TTSRequest(text="good one", language="English", instruct="calm voice"),
        TTSRequest(text="flaky", language="English", instruct="calm voice"),
    ]

    results = await service.generate_batch(requests)

    assert isinstance(results[0], bytes)
    assert isinstance(results[1], RuntimeError)
    assert str(results[1]) == "gpu hiccup"
    assert model.call_texts == [("good one", "flaky"), ("flaky",)]


def test_is_loaded_false_before_load():
    settings = Settings(_env_file=None)
    service = TTSModelService(settings)
    assert service.is_loaded() is False


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
