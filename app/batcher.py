import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

GenerateFn = Callable[[Sequence[TTSRequest]], Awaitable[list[bytes | Exception]]]


@dataclass
class _QueueItem:
    request: TTSRequest
    future: "asyncio.Future[bytes]"


class BatchWorker:
    def __init__(self, generate_fn: GenerateFn, window_ms: int, max_batch_size: int) -> None:
        self._generate_fn = generate_fn
        self._window_s = window_ms / 1000
        self._max_batch_size = max_batch_size
        self._queue: "asyncio.Queue[_QueueItem]" = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._current_batch: list[_QueueItem] = []

    async def submit(self, request: TTSRequest) -> bytes:
        future: "asyncio.Future[bytes]" = asyncio.get_running_loop().create_future()
        await self._queue.put(_QueueItem(request, future))
        return await future

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("BatchWorker is already running")
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._fail_pending(RuntimeError("worker stopped"))

    def queue_depth(self) -> int:
        return self._queue.qsize() + len(self._current_batch)

    def _fail_pending(self, exc: Exception) -> None:
        for item in self._current_batch:
            if not item.future.done():
                item.future.set_exception(exc)
        self._current_batch = []
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(exc)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item.future.done():
                continue
            batch = [item]
            self._current_batch = batch
            deadline = time.monotonic() + self._window_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if next_item.future.done():
                    continue
                batch.append(next_item)
            await self._dispatch(batch)
            self._current_batch = []

    async def _dispatch(self, batch: list[_QueueItem]) -> None:
        requests = [i.request for i in batch]
        try:
            results = await self._generate_fn(requests)
            if len(results) != len(batch):
                raise ValueError(
                    f"generate_fn returned {len(results)} result(s) for a batch of {len(batch)}"
                )
        except Exception as exc:  # noqa: BLE001 - propagate to callers via their Future
            logger.exception("Batch of %d request(s) failed", len(batch))
            for i in batch:
                if not i.future.done():
                    i.future.set_exception(exc)
            return
        for i, result in zip(batch, results):
            if i.future.done():
                continue
            if isinstance(result, BaseException):
                i.future.set_exception(result)
            else:
                i.future.set_result(result)
