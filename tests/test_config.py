from app.config import Settings


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert settings.device == "cuda:0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.min_free_vram_gb == 6.0
    assert settings.batch_window_ms == 150
    assert settings.max_batch_size == 4
    assert settings.max_new_tokens == 2048
    assert settings.request_timeout_s == 120.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("QWEN_TTS_PORT", "9000")
    monkeypatch.setenv("QWEN_TTS_MAX_BATCH_SIZE", "8")
    settings = Settings(_env_file=None)
    assert settings.port == 9000
    assert settings.max_batch_size == 8
