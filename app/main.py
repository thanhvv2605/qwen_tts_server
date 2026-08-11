import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from app import model as model_module
from app.batcher import BatchWorker
from app.config import settings
from app.jobs import ItemStatus, JobManager, job_to_dict
from app.model import TTSModelService
from app.schemas import JobSubmitRequest, TTSRequest

logger = logging.getLogger(__name__)

model_service = TTSModelService(settings)
batch_worker = BatchWorker(
    generate_fn=model_service.generate_batch,
    window_ms=settings.batch_window_ms,
    max_batch_size=settings.max_batch_size,
)


async def _job_submit(request: TTSRequest) -> bytes:
    return await asyncio.wait_for(
        batch_worker.submit(request), timeout=settings.request_timeout_s
    )


job_manager = JobManager(submit_fn=_job_submit, settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.wipe_results_dir()
    model_service.load()
    batch_worker.start()
    yield
    await job_manager.shutdown()
    await batch_worker.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/tts/voice-design")
async def voice_design(request: TTSRequest) -> Response:
    try:
        wav_bytes = await asyncio.wait_for(
            batch_worker.submit(request), timeout=settings.request_timeout_s
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Timed out waiting for GPU queue") from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 500
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health")
async def health() -> dict:
    vram_free_gb = None
    if model_service.is_loaded():
        loop = asyncio.get_running_loop()
        vram_free_gb = await loop.run_in_executor(
            None, model_module.check_vram, settings.device, settings.min_free_vram_gb
        )
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "vram_free_gb": vram_free_gb,
        "queue_depth": batch_worker.queue_depth(),
    }


@app.post("/v1/jobs", status_code=202)
async def submit_job(request: JobSubmitRequest) -> dict:
    if len(request.items) > settings.max_items_per_job:
        raise HTTPException(
            status_code=422,
            detail=f"a job may contain at most {settings.max_items_per_job} items",
        )
    job = job_manager.create_job(request.items)
    return {"job_id": job.job_id, "status": job.status.value, "total_items": len(job.items)}


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_dict(job)


@app.get("/v1/jobs/{job_id}/items/{index}/audio")
async def get_job_item_audio(job_id: str, index: int) -> Response:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if index < 0 or index >= len(job.items):
        raise HTTPException(status_code=404, detail="item index out of range")
    item = job.items[index]
    if item.status is not ItemStatus.DONE:
        raise HTTPException(status_code=409, detail=f"item not ready: {item.status.value}")
    wav_bytes = job_manager.audio_path(job_id, index).read_bytes()
    return Response(content=wav_bytes, media_type="audio/wav")


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_dict(job)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
