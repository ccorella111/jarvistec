"""'wine' model: classifies the quality of a wine (low / medium / high) from its chemistry.

Trained in notebooks/wine.ipynb (Wine Quality dataset, red and white vinho verde).
Classes: low (sensory score <= 5), medium (6), high (>= 7). Artifact: models/wine.joblib.

Input (every field is optional; missing ones are imputed with typical values):
    {"type": "red|white", "fixed_acidity": 7.0, "volatile_acidity": 0.27, "citric_acid": 0.36,
     "residual_sugar": 2.0, "chlorides": 0.045, "free_sulfur_dioxide": 30, "total_sulfur_dioxide": 120,
     "density": 0.995, "pH": 3.2, "sulphates": 0.5, "alcohol": 10.5}
Voice command: "JarvisTEC, quality of a red wine with alcohol 12 and pH 3.3"
"""
from pathlib import Path

import joblib
import numpy as np

from predictors._common import build_row, choice, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "wine.joblib"

NUM = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
       "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]
CAT = ["type"]
COLUMNS = ["type"] + NUM
CLASSES = ["low", "medium", "high"]          # order of the labels 0, 1, 2
RANGES = {  # wide limits (the training data fits comfortably inside)
    "fixed_acidity": (3, 16), "volatile_acidity": (0.05, 2), "citric_acid": (0, 1.8),
    "residual_sugar": (0.4, 70), "chlorides": (0.005, 0.7), "free_sulfur_dioxide": (1, 300),
    "total_sulfur_dioxide": (5, 450), "density": (0.98, 1.04), "pH": (2.5, 4.2),
    "sulphates": (0.2, 2.2), "alcohol": (7, 16),
}
SPECS = {"type": ("cat", ["red", "white"])} | {c: ("num", *RANGES[c]) for c in NUM}

# voice command: field -> words that introduce its value
VOICE_FIELDS = {"alcohol": ["alcohol", "abv", "alcohol content"], "pH": ["ph"],
                "residual_sugar": ["residual sugar", "sugar"], "volatile_acidity": ["volatile acidity"],
                "fixed_acidity": ["fixed acidity"], "citric_acid": ["citric acid"], "chlorides": ["chlorides"],
                "sulphates": ["sulphates", "sulfates"], "density": ["density"]}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads the wine type and some measurements from the spoken phrase."""
    t = normalize(text)
    e: dict = {}
    wine_type = choice(t, {"red": ["red wine", "red"], "white": ["white wine", "white"]})
    if wine_type:
        e["type"] = wine_type
    for field, words in VOICE_FIELDS.items():
        v = number(t, after=words)
        if v is not None:
            e[field] = v
    return e or None


META = {
    "name": "wine",
    "description": "Classifies the quality of a wine (low, medium or high) from its chemical composition",
    "keywords": ["wine quality", "quality of a wine", "wine", "red wine", "white wine"],
    "example_input": {"type": "white", "fixed_acidity": 6.8, "volatile_acidity": 0.26, "citric_acid": 0.33,
                      "residual_sugar": 5.0, "chlorides": 0.037, "free_sulfur_dioxide": 35,
                      "total_sulfur_dioxide": 130, "density": 0.9935, "pH": 3.2, "sulphates": 0.5,
                      "alcohol": 11.5},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/wine.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    row, imputed = build_row(payload, SPECS)           # missing fields -> NaN (the pipeline imputes them)
    if len(imputed) == len(COLUMNS):
        raise ValueError("Provide at least one measurement of the wine (for example 'alcohol')")
    proba = b["pipeline"].predict_proba(frame(row, SPECS))[0]
    k = int(np.argmax(proba))
    pct = round(float(proba[k]) * 100, 1)
    return {
        "prediction": f"{CLASSES[k]} quality ({pct} %)",
        "class": CLASSES[k],
        "confidence_pct": pct,
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "Estimate from physicochemical data; the real quality is defined by a sensory tasting.",
    }
