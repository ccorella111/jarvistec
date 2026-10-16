"""'hepatitis' model: classifies a patient as healthy, hepatitis C, fibrosis or cirrhosis from blood tests.

Trained in notebooks/hepatitis.ipynb (HCV data, UCI). Artifact: models/hepatitis.joblib.

Input (every field is optional; missing ones are filled with typical values):
    {"Age": 56, "Sex": "m|f", "ALB": 33.0, "ALP": 60, "ALT": 25, "AST": 92.9, "BIL": 34.0,
     "CHE": 3.4, "CHOL": 4.0, "CREA": 75, "GGT": 70, "PROT": 70}
Voice command: "JarvisTEC, which type of hepatitis does a 56 year old man with AST 92 and bilirubin 34 have?"
"""
from pathlib import Path

import joblib

from predictors._common import build_row, choice, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hepatitis.joblib"

LABS = ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
RANGES = {"Age": (15, 100), "ALB": (10, 70), "ALP": (5, 500), "ALT": (1, 400), "AST": (5, 500), "BIL": (0.5, 300),
          "CHE": (1, 20), "CHOL": (1, 10), "CREA": (10, 1100), "GGT": (1, 700), "PROT": (40, 100)}
SPECS = {"Age": ("num", *RANGES["Age"]), "Sex": ("cat", ["m", "f"], {"male": "m", "female": "f"})} | \
        {lab: ("num", *RANGES[lab]) for lab in LABS}
VOICE_WORDS = {"ALB": ["albumin", "alb"], "ALP": ["alp", "alkaline phosphatase"], "ALT": ["alt"], "AST": ["ast"],
               "BIL": ["bilirubin", "bil"], "CHE": ["cholinesterase", "che"], "CHOL": ["cholesterol", "chol"],
               "CREA": ["creatinine", "crea"], "GGT": ["ggt"], "PROT": ["protein", "prot"]}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads age, sex and blood-test values from the spoken phrase."""
    t = normalize(text)
    e: dict = {}
    age = number(t, after=["age", "aged"], before=["years old", "year old", "years of age"])
    if age is not None:
        e["Age"] = age
    sex = choice(t, {"m": ["man", "male", "gentleman"], "f": ["woman", "female", "lady"]})
    if sex:
        e["Sex"] = sex
    for field, words in VOICE_WORDS.items():
        v = number(t, after=words)
        if v is not None:
            e[field] = v
    return e or None


META = {
    "name": "hepatitis",
    "description": "Classifies a patient as healthy, hepatitis C, fibrosis or cirrhosis from routine blood tests",
    "keywords": ["hepatitis", "type of hepatitis", "hepatitis type", "hepatitis c"],
    "example_input": {"Age": 56, "Sex": "m", "ALB": 33.0, "AST": 92.9, "BIL": 34.0, "CHE": 3.4},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/hepatitis.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    row, imputed = build_row(payload, SPECS, b["typical_values"])
    pipeline, X = b["pipeline"], frame(row, SPECS)
    label = str(pipeline.predict(X)[0])
    out = {
        "prediction": label,
        "group": "healthy" if label == "healthy" else "liver disease",
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "Academic exercise with 608 patients: not a medical diagnosis.",
    }
    if hasattr(pipeline, "predict_proba"):
        out["probabilities"] = {str(c): round(float(p), 2) for c, p in zip(pipeline.classes_, pipeline.predict_proba(X)[0])}
    return out
