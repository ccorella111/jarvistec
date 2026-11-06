"""Facial emotion recognition that runs on this computer (no cloud, no keys, no quota).

Google Cloud Vision stopped returning emotions on 2026-10-15 (every emotion comes back UNKNOWN), so the
default provider is now HSEmotion (https://github.com/HSE-asavchenko/face-emotion-recognition, Apache-2.0):
small ONNX networks trained on AffectNet.

Pipeline: decode image -> find the largest face with OpenCV (Haar cascades) -> crop it with a margin ->
average the probabilities of two HSEmotion models -> return the most likely emotion.

The model files (about 16 MB each) are downloaded once to ~/.hsemotion the first time they are needed.
"""
import threading
import urllib.request
from pathlib import Path

import numpy as np
from starlette.concurrency import run_in_threadpool

MODEL_NAMES = ("enet_b0_8_best_vgaf", "enet_b0_8_best_afew")
MODEL_URL = ("https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/"
             "models/affectnet_emotions/onnx/{name}.onnx")
CACHE_DIR = Path.home() / ".hsemotion"
FACE_MARGIN = 0.3   
MIN_FACE = 60       
CASCADES = ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")

_lock = threading.Lock()
_recognizers: list = []
_detectors: list = []


def _download_models() -> None:
    """Fetches the model files ourselves: the library's own downloader breaks on some Python versions."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        path = CACHE_DIR / f"{name}.onnx"
        if path.exists() and path.stat().st_size > 1_000_000:
            continue
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(name=name), tmp)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded model {name} looks wrong; check your internet connection")
        tmp.replace(path)


def _load_detectors() -> list:
    import cv2

    if not _detectors:
        for name in CASCADES:
            c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
            if not c.empty():
                _detectors.append(c)
    return _detectors


def load() -> None:
    """Loads the emotion models (downloading them the first time). Called at startup and on first use."""
    with _lock:
        if _recognizers:
            return
        _download_models()
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        _recognizers.extend(HSEmotionRecognizer(model_name=n) for n in MODEL_NAMES)


def _decode(image: bytes):
    import cv2

    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file is not a valid image (use jpg or png)")
    return img


def _find_face(img):
    """Returns (x, y, w, h) of the largest face, or None."""
    import cv2

    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for det in _load_detectors():
        faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE, MIN_FACE))
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda b: b[2] * b[3]))
    return None


def _crop(img, box, margin: float = FACE_MARGIN):
    """Face crop (RGB) with a margin, clipped to the image borders."""
    import cv2

    x, y, w, h = box
    m = int(margin * w)
    height, width = img.shape[:2]
    face = img[max(0, y - m):min(height, y + h + m), max(0, x - m):min(width, x + w + m)]
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def _classify(face_rgb) -> dict:
    """Average of the models' probabilities -> {emotion: probability}."""
    load()
    total, names = None, None
    for rec in _recognizers:
        _, scores = rec.predict_emotions(face_rgb, logits=False)
        names = rec.idx_to_class
        total = scores if total is None else total + scores
    probs = total / len(_recognizers)
    return {names[i].lower(): float(p) for i, p in enumerate(probs)}


def _analyze(image: bytes) -> dict:
    img = _decode(image)
    box = _find_face(img)
    if box is None:
        return {"faces": 0, "emotion": None, "confidence": 0.0}
    probs = _classify(_crop(img, box))
    top = max(probs, key=probs.get)
    return {"faces": 1, "emotion": top, "confidence": round(probs[top], 2),
            "details": {k: round(v, 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])},
            "provider": "local"}


async def detect_emotion(image: bytes) -> dict:
    return await run_in_threadpool(_analyze, image)
