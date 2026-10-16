"""'cirrhosis' model: estimates the stage of a patient's liver disease (stage 1-2, 3 or 4).

Trained in notebooks/cirrhosis.ipynb (Mayo Clinic primary biliary cirrhosis study).
Artifact: models/cirrhosis.joblib.

Input (every field is optional; missing ones are filled with typical values):
    {"Age": 45 (years), "Sex": "F|M", "Ascites": "N|Y", "Hepatomegaly": "N|Y", "Spiders": "N|Y",
     "Edema": "N|S|Y", "Bilirubin": 0.7, "Cholesterol": 300, "Albumin": 3.9, "Copper": 70,
     "Alk_Phos": 1500, "SGOT": 100, "Tryglicerides": 110, "Platelets": 250, "Prothrombin": 10.5}
Voice command: "JarvisTEC, which cirrhosis stage is a 60 year old woman with ascites and bilirubin 6?"
"""
from pathlib import Path

import joblib

from predictors._common import build_row, choice, flag, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cirrhosis.joblib"

NUMERIC = ["Age", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides",
           "Platelets", "Prothrombin"]
CATEGORICAL = ["Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
RANGES = {"Age": (18, 100), "Bilirubin": (0.1, 30), "Cholesterol": (50, 2000), "Albumin": (1, 6),
          "Copper": (1, 700), "Alk_Phos": (100, 20000), "SGOT": (10, 500), "Tryglicerides": (20, 700),
          "Platelets": (20, 800), "Prothrombin": (8, 25)}
SPECS_BASE = {"Age": ("num", *RANGES["Age"]), "Sex": ("cat", ["F", "M"], {"female": "F", "male": "M"})}
VOICE_NUMBERS = {"Bilirubin": ["bilirubin"], "Albumin": ["albumin"], "Platelets": ["platelets"],
                 "Prothrombin": ["prothrombin", "prothrombin time"], "Cholesterol": ["cholesterol"],
                 "Copper": ["copper"]}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads age, sex, clinical signs and some lab values from the phrase."""
    t = normalize(text)
    e: dict = {}
    age = number(t, after=["age", "aged"], before=["years old", "year old", "years of age"])
    if age is not None:
        e["Age"] = age
    sex = choice(t, {"M": ["man", "male", "gentleman"], "F": ["woman", "female", "lady"]})
    if sex:
        e["Sex"] = sex
    for field, phrases in (("Ascites", ["ascites"]), ("Hepatomegaly", ["hepatomegaly", "enlarged liver"]),
                           ("Spiders", ["spiders", "spider angiomas", "spider nevi"]), ("Edema", ["edema", "swelling"])):
        v = flag(t, phrases)
        if v is not None:
            e[field] = "Y" if v else "N"
    for field, words in VOICE_NUMBERS.items():
        v = number(t, after=words)
        if v is not None:
            e[field] = v
    return e or None


META = {
    "name": "cirrhosis",
    "description": "Estimates the stage of a patient's liver disease (primary biliary cirrhosis)",
    "keywords": ["cirrhosis", "cirrhosis stage", "liver stage", "liver cirrhosis"],
    "example_input": {"Age": 45, "Sex": "F", "Ascites": "N", "Hepatomegaly": "N", "Spiders": "N", "Edema": "N",
                      "Bilirubin": 0.7, "Albumin": 3.9, "Platelets": 300, "Prothrombin": 10.0},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/cirrhosis.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    specs = {f: ("cat", b["categories"][f]) if f in CATEGORICAL else ("num", *RANGES[f]) for f in b["features"]}
    specs["Sex"] = ("cat", b["categories"]["Sex"], SPECS_BASE["Sex"][2])
    row, imputed = build_row(payload, specs, b["typical_values"])
    pipeline = b["pipeline"]
    proba = pipeline.predict_proba(frame(row, specs))[0]
    by_class = dict(zip(pipeline.classes_, proba))
    best = max(by_class, key=by_class.get)
    return {
        "prediction": str(best),
        "probabilities": {c: round(float(by_class[c]), 2) for c in b["classes"]},
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "Indicative only: the real stage is determined by a biopsy. Not a medical diagnosis.",
    }
