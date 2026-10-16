from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import google_vision

router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("")
async def emotion(image: UploadFile = File(...)):
    """Receives a photo or camera frame (jpg/png) and returns the dominant emotion."""
    try:
        return await google_vision.detect_emotion(await image.read())
    except Exception as e:
        raise HTTPException(502, f"Google Vision error: {e}")
