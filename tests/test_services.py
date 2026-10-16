from types import SimpleNamespace

from app.services.azure_stt import _parse
from app.services.google_vision import _summarize


def test_vision_joy():
    f = SimpleNamespace(joy_likelihood=5, surprise_likelihood=1, anger_likelihood=1, sorrow_likelihood=2)
    r = _summarize(f)
    assert r["emotion"] == "happiness" and r["confidence"] == 1.0


def test_vision_inferred_neutral():
    f = SimpleNamespace(joy_likelihood=1, surprise_likelihood=1, anger_likelihood=2, sorrow_likelihood=1)
    assert _summarize(f)["emotion"] == "neutral"


def test_azure_parsing():
    ok = {"RecognitionStatus": "Success", "NBest": [{"Display": "Hello.", "Confidence": 0.9}]}
    assert _parse(ok) == {"text": "Hello.", "confidence": 0.9}
    assert _parse({"RecognitionStatus": "NoMatch"})["text"] == ""
