"""'stroke' model: estimates a patient's risk of stroke.

Trained in notebooks/stroke.ipynb. Artifact: models/stroke.joblib (dict with the calibrated
sklearn model, the alert threshold and the dataset prevalence).

Input (only 'age' is required; missing fields are imputed):
    {"gender": "Male|Female|Other", "age": 70, "hypertension": 0|1, "heart_disease": 0|1,
     "ever_married": "Yes|No", "work_type": "Private|Self-employed|Govt_job|children|Never_worked",
     "Residence_type": "Urban|Rural", "avg_glucose_level": 105.9, "bmi": 28.1,
     "smoking_status": "formerly smoked|never smoked|smokes|Unknown"}
Voice command: "JarvisTEC, stroke risk of a 70 year old man with hypertension"

WARNING: academic project, not a medical diagnosis tool.
"""
from pathlib import Path

import joblib

from predictors._common import build_row, choice, flag, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stroke.joblib"

NUM = ["age", "avg_glucose_level", "bmi"]
BIN = ["hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
COLUMNS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
           "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
CATEGORIES = {
    "gender": ["Male", "Female", "Other"],
    "ever_married": ["Yes", "No"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
}
RANGES = {"age": (0, 120), "avg_glucose_level": (30, 500), "bmi": (10, 100)}
SPECS = {
    "gender": ("cat", CATEGORIES["gender"], {"man": "Male", "woman": "Female"}),
    "age": ("num", *RANGES["age"]),
    "hypertension": ("bin",), "heart_disease": ("bin",),
    "ever_married": ("cat", CATEGORIES["ever_married"]),
    "work_type": ("cat", CATEGORIES["work_type"]),
    "Residence_type": ("cat", CATEGORIES["Residence_type"]),
    "avg_glucose_level": ("num", *RANGES["avg_glucose_level"]),
    "bmi": ("num", *RANGES["bmi"]),
    "smoking_status": ("cat", CATEGORIES["smoking_status"], {"smoker": "smokes", "former smoker": "formerly smoked"}),
}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads age, sex and risk factors from the spoken phrase."""
    t = normalize(text)
    e: dict = {}
    age = number(t, after=["age", "aged"], before=["years old", "year old", "years of age", "yo"])
    if age is not None:
        e["age"] = age
    gender = choice(t, {"Male": ["man", "male", "boy", "gentleman"], "Female": ["woman", "female", "girl", "lady"]})
    if gender:
        e["gender"] = gender
    for field, phrases in (("hypertension", ["hypertension", "high blood pressure", "hypertensive"]),
                           ("heart_disease", ["heart disease", "heart problems", "heart condition", "cardiac disease"])):
        v = flag(t, phrases)
        if v is not None:
            e[field] = v
    glucose = number(t, after=["glucose", "glucose level", "blood sugar"])
    if glucose is not None:
        e["avg_glucose_level"] = glucose
    bmi = number(t, after=["bmi", "body mass index"])
    if bmi is not None:
        e["bmi"] = bmi
    smoking = choice(t, {
        "never smoked": ["does not smoke", "doesn t smoke", "non smoker", "nonsmoker", "never smoked", "not a smoker"],
        "formerly smoked": ["former smoker", "used to smoke", "quit smoking", "ex smoker", "formerly smoked", "stopped smoking"],
        "smokes": ["smoker", "smokes", "smoking", "smoke"]})
    if smoking:
        e["smoking_status"] = smoking
    return e or None


META = {
    "name": "stroke",
    "description": "Estimates a patient's risk of stroke (cerebrovascular accident)",
    "keywords": ["stroke", "stroke risk", "cerebrovascular", "brain stroke", "cerebral stroke"],
    "example_input": {"gender": "Male", "age": 70, "hypertension": 1, "heart_disease": 0,
                      "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
                      "avg_glucose_level": 105.9, "bmi": 28.1, "smoking_status": "formerly smoked"},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/stroke.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    row, imputed = build_row(payload, SPECS)           # missing fields -> NaN (the pipeline imputes them)
    if "age" in imputed:
        raise ValueError("'age' (in years) is missing: it is the most important input of the model")
    p = float(b["pipeline"].predict_proba(frame(row, SPECS))[0, 1])

    threshold, prevalence = b["threshold"], b["prevalence"]
    level = "high" if p >= threshold else ("moderate" if p >= prevalence else "low")
    pct = round(p * 100, 1)
    return {
        "prediction": f"{level} risk ({pct} %)",
        "risk_level": level,
        "probability": round(p, 4),
        "probability_pct": pct,
        "alert": bool(p >= threshold),
        "alert_threshold": round(threshold, 4),
        "dataset_prevalence": round(prevalence, 4),
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "Academic project; not a medical diagnosis. Consult a healthcare professional.",
    }
