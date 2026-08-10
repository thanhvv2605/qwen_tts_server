import asyncio
import io

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from app import main as main_module


def _fake_wav_bytes() -> bytes:
    wav = np.zeros(2400, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    # Module-scoped (rather than the default function scope) so the app's
    # lifespan starts and stops exactly once for this file. batch_worker is a
    # process-lifetime singleton whose internal asyncio.Queue binds to
    # whichever event loop first uses it; TestClient spins up a fresh event
    # loop on every `with TestClient(...)` entry, so re-entering it per-test
    # would try to reuse that queue across different event loops and crash
    # with "Queue ... is bound to a different event loop". A single
    # start/stop cycle for the whole module matches how the real server
    # actually runs (lifespan started once per process) and sidesteps that.
    mp = pytest.MonkeyPatch()
    mp.setattr(main_module.model_service, "load", lambda: None)
    mp.setattr(main_module.model_service, "is_loaded", lambda: True)
    mp.setattr("app.model.check_vram", lambda device, min_free_gb: 20.0)

    async def fake_generate_fn(requests):
        return [_fake_wav_bytes() for _ in requests]

    mp.setattr(main_module.batch_worker, "_generate_fn", fake_generate_fn)

    with TestClient(main_module.app) as test_client:
        yield test_client

    mp.undo()


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
