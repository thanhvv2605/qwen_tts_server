import asyncio

import pytest

from app.batcher import BatchWorker, _QueueItem
from app.schemas import TTSRequest


def _req(text: str) -> TTSRequest:
    return TTSRequest(text=text, language="English", instruct="calm voice")


async def test_single_request_is_processed():
    calls = []

    async def fake_generate(requests):
        calls.append(list(requests))
        return [f"audio-for-{r.text}".encode() for r in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=50, max_batch_size=4)
    worker.start()
    try:
        result = await worker.submit(_req("hello"))
    finally:
        await worker.stop()

    assert result == b"audio-for-hello"
    assert calls == [[_req("hello")]]


async def test_requests_within_window_are_batched_together():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=100, max_batch_size=4)
    worker.start()
    try:
        task_a = asyncio.create_task(worker.submit(_req("a")))
        await asyncio.sleep(0.02)
        task_b = asyncio.create_task(worker.submit(_req("b")))
        await task_a
        await task_b
    finally:
        await worker.stop()

    assert calls == [2]


async def test_requests_outside_window_are_separate_batches():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=30, max_batch_size=4)
    worker.start()
    try:
        await worker.submit(_req("a"))
        await worker.submit(_req("b"))
    finally:
        await worker.stop()

    assert calls == [1, 1]


async def test_batch_caps_at_max_batch_size():
    calls = []

    async def fake_generate(requests):
        calls.append(len(requests))
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=200, max_batch_size=2)
    worker.start()
    try:
        tasks = [asyncio.create_task(worker.submit(_req(str(i)))) for i in range(3)]
        await asyncio.gather(*tasks)
    finally:
        await worker.stop()

    assert calls == [2, 1]


async def test_batch_error_propagates_to_all_pending_futures():
    async def failing_generate(requests):
        raise RuntimeError("boom")

    worker = BatchWorker(generate_fn=failing_generate, window_ms=50, max_batch_size=4)
    worker.start()
    try:
        task_a = asyncio.create_task(worker.submit(_req("a")))
        await asyncio.sleep(0.01)
        task_b = asyncio.create_task(worker.submit(_req("b")))

        with pytest.raises(RuntimeError, match="boom"):
            await task_a
        with pytest.raises(RuntimeError, match="boom"):
            await task_b
    finally:
        await worker.stop()


async def test_start_twice_raises():
    async def fake_generate(requests):
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=50, max_batch_size=4)
    worker.start()
    try:
        with pytest.raises(RuntimeError):
            worker.start()
    finally:
        await worker.stop()


async def test_stop_fails_pending_requests_instead_of_hanging():
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_generate(requests):
        started.set()
        await release.wait()
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=slow_generate, window_ms=200, max_batch_size=4)
    worker.start()

    in_flight = asyncio.create_task(worker.submit(_req("in-flight")))
    await started.wait()
    queued = asyncio.create_task(worker.submit(_req("queued")))
    await asyncio.sleep(0.01)  # let "queued" land in the queue behind the in-flight batch

    await worker.stop()

    with pytest.raises(RuntimeError, match="worker stopped"):
        await in_flight
    with pytest.raises(RuntimeError, match="worker stopped"):
        await queued


async def test_mismatched_result_length_fails_batch_without_killing_worker():
    async def bad_generate(requests):
        return [b"x"]  # wrong length for a batch of 2

    worker = BatchWorker(generate_fn=bad_generate, window_ms=100, max_batch_size=4)
    worker.start()
    try:
        task_a = asyncio.create_task(worker.submit(_req("a")))
        await asyncio.sleep(0.01)
        task_b = asyncio.create_task(worker.submit(_req("b")))

        with pytest.raises(ValueError):
            await task_a
        with pytest.raises(ValueError):
            await task_b

        # worker must still be alive for subsequent requests
        async def good_generate(requests):
            return [b"ok" for _ in requests]

        worker._generate_fn = good_generate
        result = await worker.submit(_req("c"))
        assert result == b"ok"
    finally:
        await worker.stop()


async def test_already_done_item_is_skipped_without_wasting_a_batch_slot():
    calls = []

    async def fake_generate(requests):
        calls.append([r.text for r in requests])
        return [b"x" for _ in requests]

    worker = BatchWorker(generate_fn=fake_generate, window_ms=50, max_batch_size=2)

    dead_future = asyncio.get_event_loop().create_future()
    dead_future.cancel()
    worker._queue.put_nowait(_QueueItem(_req("dead"), dead_future))

    worker.start()
    try:
        result = await worker.submit(_req("alive"))
    finally:
        await worker.stop()

    assert result == b"x"
    assert all("dead" not in batch for batch in calls)


async def test_mixed_success_and_failure_results_resolve_independently():
    async def mixed_generate(requests):
        results = []
        for r in requests:
            if r.text == "bad":
                results.append(RuntimeError("audio self-check failed"))
            else:
                results.append(f"audio-for-{r.text}".encode())
        return results

    worker = BatchWorker(generate_fn=mixed_generate, window_ms=100, max_batch_size=4)
    worker.start()
    try:
        task_good1 = asyncio.create_task(worker.submit(_req("good1")))
        task_bad = asyncio.create_task(worker.submit(_req("bad")))
        task_good2 = asyncio.create_task(worker.submit(_req("good2")))
        await asyncio.sleep(0.01)

        good1_result = await task_good1
        good2_result = await task_good2
        with pytest.raises(RuntimeError, match="audio self-check failed"):
            await task_bad
    finally:
        await worker.stop()

    assert good1_result == b"audio-for-good1"
    assert good2_result == b"audio-for-good2"
