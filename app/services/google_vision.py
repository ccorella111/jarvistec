"""Facial emotion with Google Cloud Vision (face detection).

DEPRECATED BY GOOGLE (2026-10-15): face detection now returns UNKNOWN for every emotion, so this
provider is kept only for reference. Use EMOTION_PROVIDER=local (app/services/local_emotion.py).

Vision does NOT return percentages: it returns a likelihood level (VERY_UNLIKELY..VERY_LIKELY) for
4 emotions: joy, surprise, anger and sorrow. "neutral" is inferred when none reaches POSSIBLE.
Names are returned in Azure style (happiness, sadness, anger, surprise, neutral).
"""
from starlette.concurrency import run_in_threadpool

from app.config import settings

# Likelihood enum: 0 UNKNOWN, 1 VERY_UNLIKELY, 2 UNLIKELY, 3 POSSIBLE, 4 LIKELY, 5 VERY_LIKELY
LEVEL = {0: 0.0, 1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
FIELDS = [("joy_likelihood", "happiness"), ("surprise_likelihood", "surprise"),
          ("anger_likelihood", "anger"), ("sorrow_likelihood", "sadness")]

_client = None


def _summarize(face) -> dict:
    if all(int(getattr(face, field)) == 0 for field, _ in FIELDS):
        return {"emotion": None, "confidence": 0.0, "unavailable": True,
                "note": "Google Vision no longer returns emotions (deprecated 2026-10-15)"}
    details = {name: LEVEL[int(getattr(face, field))] for field, name in FIELDS}
    top = max(details, key=details.get)
    if details[top] < 0.5:
        return {"emotion": "neutral", "confidence": round(1 - details[top], 2), "details": details,
                "note": "neutral is inferred: no emotion reached 'possible'"}
    return {"emotion": top, "confidence": details[top], "details": details}


def _call(image: bytes):
    global _client
    from google.cloud import vision  # lazy import: not needed in mock mode

    if _client is None:
        _client = vision.ImageAnnotatorClient()
    resp = _client.face_detection(image=vision.Image(content=image))
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.face_annotations


async def detect_emotion(image: bytes) -> dict:
    if settings.EMOTION_PROVIDER == "mock":
        return {"faces": 1, "emotion": "happiness", "confidence": 0.75, "mock": True}

    faces = await run_in_threadpool(_call, image)
    if not faces:
        return {"faces": 0, "emotion": None, "confidence": 0.0}
    return {"faces": len(faces), **_summarize(faces[0])}
