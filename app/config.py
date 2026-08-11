from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN_TTS_")

    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    min_free_vram_gb: float = 6.0
    batch_window_ms: int = 150
    max_batch_size: int = 4
    max_new_tokens: int = 2048
    request_timeout_s: float = 120.0
    max_plausible_words_per_second: float = 4.5
    audio_self_check_max_retries: int = 2
    audio_self_check_enabled: bool = True


settings = Settings()
