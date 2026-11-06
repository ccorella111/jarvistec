"""Voz a texto con Google Cloud Speech-to-Text"""
from app.config import settings


def transcribir(audio: bytes, content_type: str = "") -> dict:
    if settings.USE_MOCK:
        return {"texto": "jarvistec calidad del vino", "confianza": 0.95, "mock": True}

    from google.cloud import speech  

    client = speech.SpeechClient()
    if "webm" in content_type:      
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=settings.SPEECH_LANGUAGE,
        )
    else:                           
        config = speech.RecognitionConfig(language_code=settings.SPEECH_LANGUAGE)

    resp = client.recognize(config=config, audio=speech.RecognitionAudio(content=audio))
    if not resp.results:
        return {"texto": "", "confianza": 0.0}
    alt = resp.results[0].alternatives[0]
    return {"texto": alt.transcript, "confianza": alt.confidence}
