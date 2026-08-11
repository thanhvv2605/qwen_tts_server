import io
import threading
import time

import soundfile as sf

from app import main as main_module


def _item(text: str) -> dict:
    return {"text": text, "language": "English", "instruct": "calm voice"}


def _wait_until_finished(client, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] not in ("pending", "running"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s: {body}")


def test_startup_wiped_stale_results(client, results_dir):
    assert not (results_dir / "stale.txt").exists()


def test_submit_poll_download_roundtrip(client):
    resp = client.post("/v1/jobs", json={"items": [_item("hello"), _item("world")]})
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"].startswith("j_")
    assert body["total_items"] == 2

    final = _wait_until_finished(client, body["job_id"])
    assert final["status"] == "completed"
    assert final["done"] == 2
    assert final["failed"] == 0
    assert final["items"][0]["status"] == "done"
    assert final["items"][1]["status"] == "done"

    audio = client.get(f"/v1/jobs/{body['job_id']}/items/0/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(audio.content))
    assert sr == 24000


def test_submit_rejects_empty_items(client):
    resp = client.post("/v1/jobs", json={"items": []})
    assert resp.status_code == 422


def test_submit_rejects_invalid_item(client):
    resp = client.post(
        "/v1/jobs",
        json={"items": [_item("ok"), {"text": "", "language": "English", "instruct": "x"}]},
    )
    assert resp.status_code == 422


def test_submit_rejects_too_many_items(client, monkeypatch):
    monkeypatch.setattr(main_module.settings, "max_items_per_job", 2)
    resp = client.post("/v1/jobs", json={"items": [_item("a"), _item("b"), _item("c")]})
    assert resp.status_code == 422
    assert "at most 2" in resp.json()["detail"]


def test_get_unknown_job_returns_404(client):
    resp = client.get("/v1/jobs/j_doesnotexist")
    assert resp.status_code == 404


def test_download_unfinished_item_returns_409(client, monkeypatch):
    import asyncio

    # A cooperative, bounded gate rather than a fixed sleep: even if the
    # batcher's collection window closes and captures this fn into an
    # in-flight dispatch before monkeypatch teardown runs, setting the
    # threading.Event below lets that dispatch complete promptly instead of
    # stalling the shared, process-lifetime batch_worker for a fixed delay.
    release = threading.Event()

    async def gated_generate_fn(requests):
        while not release.is_set():
            await asyncio.sleep(0.01)
        from tests.conftest import fake_wav_bytes

        return [fake_wav_bytes() for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", gated_generate_fn)

    resp = client.post("/v1/jobs", json={"items": [_item("slow")]})
    job_id = resp.json()["job_id"]

    audio = client.get(f"/v1/jobs/{job_id}/items/0/audio")
    assert audio.status_code == 409
    assert "item not ready" in audio.json()["detail"]

    out_of_range = client.get(f"/v1/jobs/{job_id}/items/5/audio")
    assert out_of_range.status_code == 404

    # Unblock the gated generation so nothing lingers into later tests, then
    # let the 1-item job finish cleanly.
    release.set()
    _wait_until_finished(client, job_id)


def test_cancel_job_and_idempotency(client, monkeypatch):
    import asyncio

    async def slow_generate_fn(requests):
        await asyncio.sleep(0.2)
        from tests.conftest import fake_wav_bytes

        return [fake_wav_bytes() for _ in requests]

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", slow_generate_fn)

    resp = client.post("/v1/jobs", json={"items": [_item(str(i)) for i in range(10)]})
    job_id = resp.json()["job_id"]

    cancel = client.delete(f"/v1/jobs/{job_id}")
    assert cancel.status_code == 200

    final = _wait_until_finished(client, job_id)
    assert final["status"] == "cancelled"
    assert any(item["status"] == "cancelled" for item in final["items"])

    again = client.delete(f"/v1/jobs/{job_id}")
    assert again.status_code == 200
    assert again.json()["status"] == "cancelled"


def test_cancel_unknown_job_returns_404(client):
    resp = client.delete("/v1/jobs/j_doesnotexist")
    assert resp.status_code == 404


def test_per_item_failure_surfaces_through_poll_and_download(client, monkeypatch):
    async def mixed_generate_fn(requests):
        from tests.conftest import fake_wav_bytes

        results = []
        for r in requests:
            if r.text == "doomed":
                results.append(RuntimeError("audio self-check failed after 2 retries"))
            else:
                results.append(fake_wav_bytes())
        return results

    monkeypatch.setattr(main_module.batch_worker, "_generate_fn", mixed_generate_fn)

    resp = client.post("/v1/jobs", json={"items": [_item("fine"), _item("doomed")]})
    job_id = resp.json()["job_id"]

    final = _wait_until_finished(client, job_id)
    assert final["status"] == "completed_with_errors"
    assert final["done"] == 1
    assert final["failed"] == 1
    assert final["items"][0]["status"] == "done"
    assert final["items"][1]["status"] == "failed"
    assert "audio self-check failed" in final["items"][1]["error"]

    ok_audio = client.get(f"/v1/jobs/{job_id}/items/0/audio")
    assert ok_audio.status_code == 200

    failed_audio = client.get(f"/v1/jobs/{job_id}/items/1/audio")
    assert failed_audio.status_code == 409
    assert failed_audio.json()["detail"] == "item not ready: failed"


def test_lifespan_shuts_down_jobs_before_batcher():
    # The lifespan must cancel job runners BEFORE the batcher fails pending
    # futures: a runner still awaiting submit() when the batcher drains would
    # see RuntimeError("worker stopped") and mark items failed instead of
    # cancelled. Pin the call order.
    import asyncio
    import unittest.mock as mock

    calls = []

    async def fake_jm_shutdown():
        calls.append("job_manager.shutdown")

    async def fake_bw_stop():
        calls.append("batch_worker.stop")

    with mock.patch.object(main_module.job_manager, "shutdown", fake_jm_shutdown), \
         mock.patch.object(main_module.batch_worker, "stop", fake_bw_stop), \
         mock.patch.object(main_module.model_service, "load", lambda: None), \
         mock.patch.object(main_module.batch_worker, "start", lambda: None), \
         mock.patch.object(main_module.job_manager, "wipe_results_dir", lambda: None):

        async def run_lifespan():
            async with main_module.lifespan(main_module.app):
                pass

        asyncio.run(run_lifespan())

    assert calls == ["job_manager.shutdown", "batch_worker.stop"]
