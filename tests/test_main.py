import asyncio
import io

import soundfile as sf

from app import main as main_module
from tests.conftest import fake_wav_bytes as _fake_wav_bytes


def test_voice_design_returns_wav(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "instruct": "calm male voice"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 24000


def test_voice_design_rejects_empty_text(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "", "language": "English", "instruct": "calm"},
    )
    assert resp.status_code == 422


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["vram_free_gb"] == 20.0
    assert body["queue_depth"] == 0


def test_voice_design_returns_500_on_generation_error(client, monkeypatch):
    async def failing_generate_fn(requests):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", failing_generate_fn)

    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "instruct": "calm voice"},
    )
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_voice_design_returns_504_on_timeout(client, monkeypatch):
    async def slow_generate_fn(requests):
        await asyncio.sleep(0.3)
        return [_fake_wav_bytes() for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", slow_generate_fn)
    monkeypatch.setattr(main_module.settings, "request_timeout_s", 0.02)

    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "instruct": "calm voice"},
    )
    assert resp.status_code == 504
