import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from app import model as model_module
from app.batcher import BatchWorker
from app.config import settings
from app.jobs import ItemStatus, JobManager, job_to_dict
from app.model import TTSModelService
from app.schemas import JobSubmitRequest, TTSRequest
from app.voices import DuplicateVoiceError, InvalidVoiceError, VoiceRegistry

logger = logging.getLogger(__name__)

voice_registry = VoiceRegistry(settings)
model_service = TTSModelService(settings, voice_registry)
batch_worker = BatchWorker(
    generate_fn=model_service.generate_batch,
    window_ms=settings.batch_window_ms,
    max_batch_size=settings.max_batch_size,
)


async def _job_submit(request: TTSRequest) -> bytes:
    try:
        return await asyncio.wait_for(
            batch_worker.submit(request), timeout=settings.request_timeout_s
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"timed out waiting for GPU queue after {settings.request_timeout_s:.0f}s"
        ) from None


job_manager = JobManager(submit_fn=_job_submit, settings=settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    voice_registry.scan()
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
    if model_service.is_loaded() or model_service.clone_is_loaded():
        loop = asyncio.get_running_loop()
        vram_free_gb = await loop.run_in_executor(
            None, model_module.check_vram, settings.device, settings.min_free_vram_gb
        )
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "clone_model_loaded": model_service.clone_is_loaded(),
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
async def get_job_item_audio(job_id: str, index: int) -> FileResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if index < 0 or index >= len(job.items):
        raise HTTPException(status_code=404, detail="item index out of range")
    item = job.items[index]
    if item.status is not ItemStatus.DONE:
        raise HTTPException(status_code=409, detail=f"item not ready: {item.status.value}")
    path = job_manager.audio_path(job_id, index)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result file missing")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job_to_dict(job)


@app.post("/v1/voices", status_code=201)
async def register_voice(
    name: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
) -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    audio_bytes = await ref_audio.read()
    loop = asyncio.get_running_loop()
    try:
        info = await loop.run_in_executor(
            None, voice_registry.register, name, audio_bytes, ref_text
        )
    except DuplicateVoiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidVoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_service.invalidate_clone_prompt(info.voice_id)
    return {"voice_id": info.voice_id, "duration_s": round(info.duration_s, 1)}


@app.get("/v1/voices")
async def list_voices() -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    return {
        "voices": [
            {
                "voice_id": v.voice_id,
                "duration_s": round(v.duration_s, 1),
                "ref_text": v.ref_text,
            }
            for v in voice_registry.list_voices()
        ]
    }


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str) -> dict:
    if not settings.voice_clone_enabled:
        raise HTTPException(status_code=503, detail="voice cloning is disabled")
    loop = asyncio.get_running_loop()
    removed = await loop.run_in_executor(None, voice_registry.delete, voice_id)
    if not removed:
        raise HTTPException(status_code=404, detail="voice not found")
    model_service.invalidate_clone_prompt(voice_id)
    return {"deleted": voice_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
