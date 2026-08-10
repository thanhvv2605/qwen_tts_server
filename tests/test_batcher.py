import asyncio

import pytest

from app.batcher import BatchWorker
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
