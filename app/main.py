from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.registry import REGISTRY, load_models
from app.routers import command, emotion, models, voice


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
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
