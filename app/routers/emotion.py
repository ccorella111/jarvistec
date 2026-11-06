from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services import emotion as emotion_service

router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("")
async def emotion(image: UploadFile = File(...)):
    """Receives a photo or camera frame (jpg/png) and returns the dominant emotion."""
    try:
        return await emotion_service.detect_emotion(await image.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Emotion service error: {e}")
