import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from app.config import Settings
from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

SubmitFn = Callable[[TTSRequest], Awaitable[bytes]]


class ItemStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    CANCELLED = "cancelled"


@dataclass
class JobItem:
    request: TTSRequest
    status: ItemStatus = ItemStatus.PENDING
    error: str | None = None


@dataclass
class Job:
    job_id: str
    items: list[JobItem]
    status: JobStatus = JobStatus.PENDING
    cancel_requested: bool = False
    task: "asyncio.Task | None" = None


def job_to_dict(job: Job) -> dict:
    items = []
    for index, item in enumerate(job.items):
        entry: dict = {"index": index, "status": item.status.value}
        if item.error is not None:
            entry["error"] = item.error
        items.append(entry)
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "total_items": len(job.items),
        "done": sum(1 for i in job.items if i.status is ItemStatus.DONE),
        "failed": sum(1 for i in job.items if i.status is ItemStatus.FAILED),
        "items": items,
    }


class JobManager:
    def __init__(self, submit_fn: SubmitFn, settings: Settings) -> None:
        self._submit_fn = submit_fn
        self._settings = settings
        self._jobs: dict[str, Job] = {}

    def _results_root(self) -> Path:
        return Path(self._settings.results_dir)

    def wipe_results_dir(self) -> None:
        shutil.rmtree(self._results_root(), ignore_errors=True)
        self._results_root().mkdir(parents=True, exist_ok=True)

    def audio_path(self, job_id: str, index: int) -> Path:
        return self._results_root() / job_id / f"{index}.wav"

    def create_job(self, requests: list[TTSRequest]) -> Job:
        job_id = f"j_{uuid.uuid4().hex[:12]}"
        job = Job(job_id=job_id, items=[JobItem(request=r) for r in requests])
        self._jobs[job_id] = job
        (self._results_root() / job_id).mkdir(parents=True, exist_ok=True)
        job.task = asyncio.create_task(self._run_job(job))
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            job.cancel_requested = True
        return job

    async def shutdown(self) -> None:
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.task.cancel()
        for job in self._jobs.values():
            if job.task is not None:
                try:
                    await job.task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - a dead runner must not abort shutdown
                    logger.exception("job %s runner died with unexpected error", job.job_id)

    async def _run_job(self, job: Job) -> None:
        try:
            await self._run_job_inner(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - runner must always leave the job terminal
            logger.exception("job %s runner crashed", job.job_id)
            for item in job.items:
                if item.status in (ItemStatus.PENDING, ItemStatus.RUNNING):
                    item.status = ItemStatus.FAILED
                    item.error = f"job runner crashed: {str(exc) or type(exc).__name__}"
            job.status = JobStatus.COMPLETED_WITH_ERRORS

    async def _run_job_inner(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        semaphore = asyncio.Semaphore(self._settings.max_batch_size)

        async def run_item(index: int, item: JobItem) -> None:
            async with semaphore:
                if job.cancel_requested:
                    item.status = ItemStatus.CANCELLED
                    return
                item.status = ItemStatus.RUNNING
                try:
                    wav_bytes = await self._submit_fn(item.request)
                    path = self.audio_path(job.job_id, index)
                    await asyncio.get_running_loop().run_in_executor(
                        None, path.write_bytes, wav_bytes
                    )
                except asyncio.CancelledError:
                    item.status = ItemStatus.CANCELLED
                    raise
                except Exception as exc:  # noqa: BLE001 - per-item failure, job continues
                    logger.warning("job %s item %d failed: %s", job.job_id, index, exc)
                    item.status = ItemStatus.FAILED
                    item.error = str(exc) or type(exc).__name__
                else:
                    item.status = ItemStatus.DONE

        try:
            await asyncio.gather(*(run_item(i, item) for i, item in enumerate(job.items)))
        except asyncio.CancelledError:
            for item in job.items:
                if item.status in (ItemStatus.PENDING, ItemStatus.RUNNING):
                    item.status = ItemStatus.CANCELLED
            job.status = JobStatus.CANCELLED
            raise
        if job.cancel_requested:
            job.status = JobStatus.CANCELLED
        elif any(item.status is ItemStatus.FAILED for item in job.items):
            job.status = JobStatus.COMPLETED_WITH_ERRORS
        else:
            job.status = JobStatus.COMPLETED
