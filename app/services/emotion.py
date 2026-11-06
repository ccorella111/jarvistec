"""Chooses the emotion provider (EMOTION_PROVIDER in .env): local | google | mock."""
from app.config import settings
from app.services import google_vision, local_emotion


async def detect_emotion(image: bytes) -> dict:
    provider = settings.EMOTION_PROVIDER
    if provider == "mock":
        return {"faces": 1, "emotion": "happiness", "confidence": 0.75, "mock": True}
    if provider == "google":
        result = await google_vision.detect_emotion(image)
        if not result.get("unavailable"):
            return result
        result = await local_emotion.detect_emotion(image)
        result["note"] = "Google Vision returned no emotions (deprecated); the local model was used"
        return result
    return await local_emotion.detect_emotion(image)
