from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.registry import REGISTRY, load_models
from app.routers import command, emotion, models, voice
from app.services import local_emotion
from starlette.concurrency import run_in_threadpool


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    if settings.EMOTION_PROVIDER in ("local", "google"):
        try:  # download/load the emotion models now so the first request is fast
            await run_in_threadpool(local_emotion.load)
        except Exception as e:  # the server still starts; /emotion reports the error
            print(f"[emotion] could not load the local model: {e}")
    yield


app = FastAPI(title="JarvisTEC API", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (emotion, voice, command, models):
    app.include_router(r.router)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "models_loaded": len(REGISTRY),
            "voice": settings.STT_PROVIDER, "emotion": settings.EMOTION_PROVIDER}


# Web interface served by the same API: http://localhost:8000/ui/
_FRONT = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/ui", StaticFiles(directory=_FRONT, html=True), name="ui")
