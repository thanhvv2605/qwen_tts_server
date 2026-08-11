import io

import numpy as np
import soundfile as sf

from app import main as main_module


def _wav_upload(duration_s: float = 2.0) -> tuple[str, bytes, str]:
    wav = np.zeros(int(duration_s * 24000), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return ("ref.wav", buf.getvalue(), "audio/wav")


def test_register_list_delete_roundtrip(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "test_voice_a", "ref_text": "Hello reference."},
        files={"ref_audio": _wav_upload()},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["voice_id"] == "test_voice_a"
    assert body["duration_s"] == 2.0

    listed = client.get("/v1/voices").json()
    ids = [v["voice_id"] for v in listed["voices"]]
    assert "test_voice_a" in ids

    deleted = client.delete("/v1/voices/test_voice_a")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": "test_voice_a"}

    ids_after = [v["voice_id"] for v in client.get("/v1/voices").json()["voices"]]
    assert "test_voice_a" not in ids_after


def test_register_duplicate_returns_409(client):
    for expected in (201, 409):
        resp = client.post(
            "/v1/voices",
            data={"name": "test_voice_dup", "ref_text": "text"},
            files={"ref_audio": _wav_upload()},
        )
        assert resp.status_code == expected
    client.delete("/v1/voices/test_voice_dup")


def test_register_invalid_audio_returns_422(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "test_voice_bad", "ref_text": "text"},
        files={"ref_audio": ("ref.wav", b"not audio", "audio/wav")},
    )
    assert resp.status_code == 422


def test_register_bad_name_returns_422(client):
    resp = client.post(
        "/v1/voices",
        data={"name": "Bad Name!", "ref_text": "text"},
        files={"ref_audio": _wav_upload()},
    )
    assert resp.status_code == 422


def test_delete_unknown_voice_returns_404(client):
    resp = client.delete("/v1/voices/never_registered")
    assert resp.status_code == 404


def test_health_reports_clone_model(client):
    body = client.get("/health").json()
    assert body["clone_model_loaded"] is False  # load() is a no-op in tests


def test_sync_generation_accepts_voice_id(client):
    # The session fixture's fake _generate_fn bypasses the model service,
    # so this exercises schema + endpoint plumbing for voice_id requests.
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "language": "English", "voice_id": "any_voice"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_sync_generation_rejects_both_instruct_and_voice_id(client):
    resp = client.post(
        "/v1/tts/voice-design",
        json={"text": "hello", "instruct": "calm", "voice_id": "v"},
    )
    assert resp.status_code == 422


def test_job_items_accept_voice_id(client):
    import time

    resp = client.post(
        "/v1/jobs",
        json={"items": [{"text": "hello", "language": "English", "voice_id": "any_voice"}]},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] not in ("pending", "running"):
            break
        time.sleep(0.02)
    assert body["status"] == "completed"
