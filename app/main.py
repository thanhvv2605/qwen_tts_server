import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from app import model as model_module
from app.batcher import BatchWorker
from app.config import settings
from app.model import TTSModelService
from app.schemas import TTSRequest

logger = logging.getLogger(__name__)

model_service = TTSModelService(settings)
batch_worker = BatchWorker(
    generate_fn=model_service.generate_batch,
    window_ms=settings.batch_window_ms,
    max_batch_size=settings.max_batch_size,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    batch_worker.start()
    yield
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
        vram_free_gb = model_module.check_vram(settings.device, settings.min_free_vram_gb)
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "vram_free_gb": vram_free_gb,
        "queue_depth": batch_worker.queue_depth(),
    }
