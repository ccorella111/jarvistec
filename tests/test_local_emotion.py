"""Tests for the emotion service. They need OpenCV but never the downloaded models or the network."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.config import settings
from app.services import emotion, google_vision, local_emotion


def _png(shape=(240, 320, 3), value=0) -> bytes:
    ok, buf = cv2.imencode(".png", np.full(shape, value, np.uint8))
    assert ok
    return buf.tobytes()


def test_blank_image_has_no_face():
    r = asyncio.run(local_emotion.detect_emotion(_png()))
    assert r == {"faces": 0, "emotion": None, "confidence": 0.0}


def test_invalid_file_is_rejected():
    with pytest.raises(ValueError):
        asyncio.run(local_emotion.detect_emotion(b"this is not an image"))


def test_crop_stays_inside_the_image():
    img = np.zeros((100, 200, 3), np.uint8)
    face = local_emotion._crop(img, (150, 60, 50, 50), margin=0.5)   # margin would cross the borders
    assert face.shape[0] <= 100 and face.shape[1] <= 200 and face.shape[2] == 3


def test_google_unknown_means_unavailable_not_neutral():
    f = SimpleNamespace(joy_likelihood=0, surprise_likelihood=0, anger_likelihood=0, sorrow_likelihood=0)
    r = google_vision._summarize(f)
    assert r["emotion"] is None and r["unavailable"] is True


def test_google_provider_falls_back_to_local(monkeypatch):
    async def fake_google(_):
        return {"emotion": None, "confidence": 0.0, "unavailable": True}

    async def fake_local(_):
        return {"faces": 1, "emotion": "happiness", "confidence": 0.9, "provider": "local"}

    monkeypatch.setattr(settings, "EMOTION_PROVIDER", "google")
    monkeypatch.setattr(google_vision, "detect_emotion", fake_google)
    monkeypatch.setattr(local_emotion, "detect_emotion", fake_local)
    r = asyncio.run(emotion.detect_emotion(b"x"))
    assert r["emotion"] == "happiness" and "local model" in r["note"]
