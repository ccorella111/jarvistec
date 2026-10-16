from fastapi import APIRouter, File, HTTPException, UploadFile

from app.routers.command import run_command
from app.services import azure_stt

router = APIRouter(prefix="/voice", tags=["voice"])


async def _transcribe(audio: UploadFile) -> dict:
    try:
        return await azure_stt.transcribe(await audio.read(), audio.content_type or "")
    except Exception as e:
        raise HTTPException(502, f"Azure Speech error: {e}")


@router.post("")
async def voice_to_text(audio: UploadFile = File(...)):
    """WAV audio -> text."""
    return await _transcribe(audio)


@router.post("/command")
async def voice_to_command(audio: UploadFile = File(...)):
    """Full flow: audio -> text -> command -> model -> result."""
    t = await _transcribe(audio)
    return {"transcription": t, "command": run_command(t["text"])}
