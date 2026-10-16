from types import SimpleNamespace

from app.services.azure_stt import _parsear
from app.services.google_vision import _resumir


def test_vision_alegria():
    f = SimpleNamespace(joy_likelihood=5, surprise_likelihood=1, anger_likelihood=1, sorrow_likelihood=2)
    r = _resumir(f)
    assert r["emocion"] == "happiness" and r["confianza"] == 1.0


def test_vision_neutral_inferido():
    f = SimpleNamespace(joy_likelihood=1, surprise_likelihood=1, anger_likelihood=2, sorrow_likelihood=1)
    assert _resumir(f)["emocion"] == "neutral"


def test_azure_parseo():
    ok = {"RecognitionStatus": "Success", "NBest": [{"Display": "Hola.", "Confidence": 0.9}]}
    assert _parsear(ok) == {"texto": "Hola.", "confianza": 0.9}
    assert _parsear({"RecognitionStatus": "NoMatch"})["texto"] == ""
