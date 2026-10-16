"""Reconocimiento de emoción con Azure Face (primera entrega).

OJO: Microsoft retiró el atributo `emotion` para clientes nuevos (Limited Access).
Si su recurso devuelve error 403/400 con ese atributo, consulten al profesor y usen
la alternativa de Google Cloud Vision (face detection: joy/sorrow/anger/surprise).
Mientras tanto USE_MOCK=true permite desarrollar la interfaz sin claves.
"""
import httpx

from app.config import settings


async def detectar_emocion(imagen: bytes) -> dict:
    if settings.USE_MOCK:
        return {"rostros": 1, "emocion": "happiness", "confianza": 0.93, "mock": True}

    url = f"{settings.AZURE_FACE_ENDPOINT.rstrip('/')}/face/v1.0/detect"
    params = {"returnFaceAttributes": "emotion", "detectionModel": "detection_01"}
    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_FACE_KEY,
        "Content-Type": "application/octet-stream",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, params=params, headers=headers, content=imagen)
    r.raise_for_status()
    rostros = r.json()
    if not rostros:
        return {"rostros": 0, "emocion": None, "confianza": 0.0}
    emociones = rostros[0]["faceAttributes"]["emotion"]
    dominante = max(emociones, key=emociones.get)
    return {
        "rostros": len(rostros),
        "emocion": dominante,
        "confianza": emociones[dominante],
        "detalle": emociones,
    }
