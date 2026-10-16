"""Speech-to-text with Azure Speech (REST API for short audio, up to ~60 s).

Requires PCM WAV audio (the frontend already converts the recording to 16 kHz mono WAV).
"""
import io
import wave

import httpx

from app.config import settings


def _sample_rate(audio: bytes) -> int:
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getframerate()
    except Exception:
        return 16000


def _parse(d: dict) -> dict:
    if d.get("RecognitionStatus") != "Success" or not d.get("NBest"):
        return {"text": "", "confidence": 0.0, "status": d.get("RecognitionStatus")}
    best = d["NBest"][0]
    return {"text": best.get("Display") or best.get("Lexical", ""), "confidence": best.get("Confidence", 0.0)}


async def transcribe(audio: bytes, content_type: str = "") -> dict:
    if settings.STT_PROVIDER == "mock":
        return {"text": "jarvistec wine quality", "confidence": 0.95, "mock": True}

    if not settings.AZURE_SPEECH_KEY or not settings.AZURE_SPEECH_REGION:
        raise RuntimeError("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION is missing in the .env file")

    url = (f"https://{settings.AZURE_SPEECH_REGION}.stt.speech.microsoft.com"
           "/speech/recognition/conversation/cognitiveservices/v1")
    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_SPEECH_KEY,
        "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={_sample_rate(audio)}",
        "Accept": "application/json",
    }
    params = {"language": settings.SPEECH_LANGUAGE, "format": "detailed"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, params=params, headers=headers, content=audio)
    if r.status_code == 401:
        raise RuntimeError("Azure rejected the key (401): check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
    r.raise_for_status()
    return _parse(r.json())
