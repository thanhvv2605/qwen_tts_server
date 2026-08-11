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
