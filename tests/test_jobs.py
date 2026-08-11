import asyncio

import pytest

from app.config import Settings
from app.jobs import ItemStatus, JobManager, JobStatus, job_to_dict
from app.schemas import TTSRequest


def _req(text: str) -> TTSRequest:
    return TTSRequest(text=text, language="English", instruct="calm voice")


def _settings(tmp_path, max_batch_size: int = 4) -> Settings:
    return Settings(_env_file=None, results_dir=str(tmp_path / "results"), max_batch_size=max_batch_size)


async def test_happy_path_all_items_done(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return f"wav-{request.text}".encode()

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a"), _req("b"), _req("c")])
    assert job.status is JobStatus.PENDING

    await job.task

    assert job.status is JobStatus.COMPLETED
    assert all(item.status is ItemStatus.DONE for item in job.items)
    for i, text in enumerate(["a", "b", "c"]):
        assert manager.audio_path(job.job_id, i).read_bytes() == f"wav-{text}".encode()


async def test_per_item_failure_keeps_job_going(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        if request.text == "bad":
            raise RuntimeError("audio self-check failed")
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("good1"), _req("bad"), _req("good2")])
    await job.task

    assert job.status is JobStatus.COMPLETED_WITH_ERRORS
    assert job.items[0].status is ItemStatus.DONE
    assert job.items[1].status is ItemStatus.FAILED
    assert "audio self-check failed" in job.items[1].error
    assert job.items[2].status is ItemStatus.DONE
    assert manager.audio_path(job.job_id, 0).exists()
    assert not manager.audio_path(job.job_id, 1).exists()


async def test_cancel_skips_pending_items_but_keeps_done_ones(tmp_path):
    release = asyncio.Event()

    async def slow_submit(request: TTSRequest) -> bytes:
        if request.text != "first":
            await release.wait()
        return b"ok"

    # max_batch_size=1 so items run strictly one at a time.
    manager = JobManager(submit_fn=slow_submit, settings=_settings(tmp_path, max_batch_size=1))
    manager.wipe_results_dir()
    job = manager.create_job([_req("first"), _req("second"), _req("third")])

    while job.items[0].status is not ItemStatus.DONE:
        await asyncio.sleep(0.01)

    manager.cancel_job(job.job_id)
    release.set()
    await job.task

    assert job.status is JobStatus.CANCELLED
    assert job.items[0].status is ItemStatus.DONE
    assert manager.audio_path(job.job_id, 0).exists()
    # "second" may have been in flight when cancel landed (done) or not
    # (cancelled); "third" must never have run.
    assert job.items[2].status is ItemStatus.CANCELLED


async def test_in_flight_capped_at_max_batch_size(tmp_path):
    in_flight = 0
    max_seen = 0
    release = asyncio.Event()

    async def tracking_submit(request: TTSRequest) -> bytes:
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await release.wait()
        in_flight -= 1
        return b"ok"

    manager = JobManager(submit_fn=tracking_submit, settings=_settings(tmp_path, max_batch_size=2))
    manager.wipe_results_dir()
    job = manager.create_job([_req(str(i)) for i in range(6)])

    await asyncio.sleep(0.05)
    release.set()
    await job.task

    assert job.status is JobStatus.COMPLETED
    assert max_seen == 2


async def test_wipe_results_dir_removes_stale_content(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    settings = _settings(tmp_path)
    manager = JobManager(submit_fn=fake_submit, settings=settings)
    stale = tmp_path / "results" / "old_job"
    stale.mkdir(parents=True)
    (stale / "0.wav").write_bytes(b"stale")

    manager.wipe_results_dir()

    results_root = tmp_path / "results"
    assert results_root.exists()
    assert list(results_root.iterdir()) == []


async def test_shutdown_cancels_running_jobs(tmp_path):
    started = asyncio.Event()

    async def hanging_submit(request: TTSRequest) -> bytes:
        started.set()
        await asyncio.Event().wait()  # never returns
        return b"unreachable"

    manager = JobManager(submit_fn=hanging_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a")])
    await started.wait()

    await manager.shutdown()

    assert job.task.done()
    assert job.status is JobStatus.CANCELLED
    assert job.items[0].status is ItemStatus.CANCELLED


async def test_job_to_dict_shape(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        if request.text == "bad":
            raise RuntimeError("boom")
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("good"), _req("bad")])
    await job.task

    d = job_to_dict(job)
    assert d["job_id"] == job.job_id
    assert d["status"] == "completed_with_errors"
    assert d["total_items"] == 2
    assert d["done"] == 1
    assert d["failed"] == 1
    assert d["items"][0] == {"index": 0, "status": "done"}
    assert d["items"][1]["index"] == 1
    assert d["items"][1]["status"] == "failed"
    assert "boom" in d["items"][1]["error"]


async def test_get_and_cancel_unknown_job(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    assert manager.get_job("j_nope") is None
    assert manager.cancel_job("j_nope") is None


async def test_timeout_failure_has_nonempty_error(tmp_path):
    async def timeout_submit(request: TTSRequest) -> bytes:
        raise TimeoutError()

    manager = JobManager(submit_fn=timeout_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a")])
    await job.task

    assert job.items[0].status is ItemStatus.FAILED
    assert job.items[0].error == "TimeoutError"


async def test_shutdown_survives_crashed_runner(tmp_path):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    manager = JobManager(submit_fn=fake_submit, settings=_settings(tmp_path))
    manager.wipe_results_dir()
    job = manager.create_job([_req("a")])
    await job.task
    # Simulate a runner that died with a non-CancelledError exception.
    crashed = asyncio.get_running_loop().create_future()
    crashed.set_exception(ValueError("runner bug"))
    job.task = asyncio.ensure_future(crashed)

    await manager.shutdown()  # must not raise


async def test_crashed_runner_leaves_job_terminal(tmp_path, monkeypatch):
    async def fake_submit(request: TTSRequest) -> bytes:
        return b"ok"

    settings = _settings(tmp_path)
    manager = JobManager(submit_fn=fake_submit, settings=settings)
    manager.wipe_results_dir()

    async def exploding_inner(job):
        raise RuntimeError("unexpected runner bug")

    monkeypatch.setattr(manager, "_run_job_inner", exploding_inner)
    job = manager.create_job([_req("a"), _req("b")])
    await job.task

    assert job.status is JobStatus.COMPLETED_WITH_ERRORS
    assert all(item.status is ItemStatus.FAILED for item in job.items)
    assert "runner crashed" in job.items[0].error
