"""Manual smoke test: run this against a live server after `uvicorn app.main:app` is up.

Usage:
    python scripts/smoke_test.py [base_url]
"""

import io
import sys

import httpx
import soundfile as sf

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def main() -> None:
    health_resp = httpx.get(f"{BASE_URL}/health", timeout=10)
    health_resp.raise_for_status()
    print("GET /health ->", health_resp.json())
    assert health_resp.json()["model_loaded"] is True, "model did not load"

    tts_resp = httpx.post(
        f"{BASE_URL}/v1/tts/voice-design",
        json={
            "text": "Xin chao, day la mot bai kiem tra.",
            "language": "Auto",
            "instruct": "Giong nu tre, vui ve, toc do noi vua phai.",
        },
        timeout=120,
    )
    tts_resp.raise_for_status()
    assert tts_resp.headers["content-type"] == "audio/wav"

    data, sr = sf.read(io.BytesIO(tts_resp.content))
    print(f"POST /v1/tts/voice-design -> {len(data)} samples at {sr}Hz")
    assert len(data) > 0, "empty audio returned"

    with open("smoke_test_output.wav", "wb") as f:
        f.write(tts_resp.content)
    print("Saved smoke_test_output.wav — listen to it to confirm audio quality.")


if __name__ == "__main__":
    main()
