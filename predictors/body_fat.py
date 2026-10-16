"""'body_fat' model: estimates a man's body fat percentage from tape-measure measurements.

Trained in notebooks/body_fat.ipynb (Johnson, 1996). Artifact: models/body_fat.joblib.

Input (every field is optional; missing ones are filled with typical values). Weight in kg, height and
circumferences in cm, age in years. Only the measurements the model uses (see GET /models) matter:
    {"Age": 43, "Height": 178, "Abdomen": 91, "Wrist": 18.3}
Voice command: "JarvisTEC, body fat of a 43 year old man with abdomen 91 and wrist 18.3"
"""
from pathlib import Path

import joblib
import numpy as np

from predictors._common import build_row, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "body_fat.joblib"

RANGES = {"Age": (15, 100), "Weight": (30, 250), "Height": (120, 230), "Neck": (25, 60), "Chest": (60, 160),
          "Abdomen": (50, 170), "Hip": (70, 170), "Thigh": (35, 100), "Knee": (25, 60), "Ankle": (15, 40),
          "Biceps": (15, 50), "Forearm": (15, 40), "Wrist": (10, 25)}
SPECS = {f: ("num", lo, hi) for f, (lo, hi) in RANGES.items()}
VOICE_WORDS = {"Abdomen": ["abdomen", "waist"], "Wrist": ["wrist"], "Height": ["height", "tall"],
               "Weight": ["weight", "weighs"], "Neck": ["neck"], "Chest": ["chest"], "Hip": ["hip", "hips"],
               "Thigh": ["thigh"], "Biceps": ["biceps"], "Forearm": ["forearm"]}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads age and measurements from the spoken phrase."""
    t = normalize(text)
    e: dict = {}
    age = number(t, after=["age", "aged"], before=["years old", "year old", "years of age"])
    if age is not None:
        e["Age"] = age
    for field, words in VOICE_WORDS.items():
        v = number(t, after=words, before=[f"cm {w}" for w in words] if field in ("Height",) else ())
        if v is not None:
            e[field] = v
    return e or None


META = {
    "name": "body_fat",
    "description": "Estimates a man's body fat percentage from simple body measurements",
    "keywords": ["body fat", "body fat percentage", "fat percentage", "percentage of body fat"],
    "example_input": {"Age": 43, "Height": 178, "Abdomen": 91, "Wrist": 18.3},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/body_fat.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    typical = {f: b["typical_values"].get(f, np.nan) for f in SPECS}
    row, imputed = build_row(payload, SPECS, typical)
    df = frame(row, SPECS)[b["features"]]                 # the model only uses some of the measurements
    pct = float(b["pipeline"].predict(df)[0])
    return {
        "prediction": round(pct, 1),
        "unit": "% body fat",
        "used_measurements": b["features"],
        "imputed_fields": [f for f in imputed if f in b["features"]],
        "model": b["model"],
        "note": "Estimate with a typical error of 3.7 points. Data from adult men; it does not replace a clinical measurement.",
    }
