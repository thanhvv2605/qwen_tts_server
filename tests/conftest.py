import io

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from app import main as main_module


def fake_wav_bytes() -> bytes:
    wav = np.zeros(2400, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    # Pre-seed a stale file so a test can verify the startup wipe.
    d = tmp_path_factory.mktemp("results")
    (d / "stale.txt").write_text("old")
    return d


@pytest.fixture(scope="session")
def client(results_dir):
    # Session-scoped (one TestClient, one event loop, one lifespan cycle for
    # the whole test session): batch_worker's internal asyncio.Queue binds
    # permanently to the first event loop that uses it, so a second
    # TestClient anywhere in the suite would crash with "Queue ... is bound
    # to a different event loop". One shared client also matches how the
    # real server runs (lifespan started once per process).
    mp = pytest.MonkeyPatch()
    mp.setattr(main_module.settings, "results_dir", str(results_dir))
    voices_dir = results_dir.parent / "voices"
    mp.setattr(main_module.settings, "voices_dir", str(voices_dir))
    mp.setattr(main_module.model_service, "load", lambda: None)
    mp.setattr(main_module.model_service, "is_loaded", lambda: True)
    mp.setattr("app.model.check_vram", lambda device, min_free_gb: 20.0)

    async def fake_generate_fn(requests):
        return [fake_wav_bytes() for _ in requests]

    mp.setattr(main_module.batch_worker, "_generate_fn", fake_generate_fn)

    try:
        with TestClient(main_module.app) as test_client:
            yield test_client
    finally:
        mp.undo()
