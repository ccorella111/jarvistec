"""'churn' model: will a telecom customer leave the company?

Trained in notebooks/churn.ipynb (Telco Customer Churn, IBM). Artifact: models/churn.joblib.

Input (every field is optional; missing ones are filled with typical values):
    {"tenure": 2, "InternetService": "DSL|Fiber optic|No", "Contract": "Month-to-month|One year|Two year",
     "PaperlessBilling": "Yes|No", "TechSupport": "Yes|No", "OnlineSecurity": "Yes|No",
     "StreamingTV": "Yes|No", "StreamingMovies": "Yes|No", "PhoneService": "Yes|No",
     "PaymentMethod": "Electronic check|Mailed check|Bank transfer (automatic)|Credit card (automatic)"}
Voice command: "JarvisTEC, will a customer with 2 months on a month to month fiber optic plan churn?"
"""
from pathlib import Path

import joblib

from predictors._common import build_row, choice, flag, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn.joblib"


def extract_input(text: str) -> dict | None:
    """Used by /command: reads tenure, internet service, contract and support services."""
    t = normalize(text)
    e: dict = {}
    months = number(t, after=["tenure"], before=["months", "month"])
    if months is not None:
        e["tenure"] = months
    internet = choice(t, {"No": ["no internet"], "Fiber optic": ["fiber optic", "fiber", "fibre"], "DSL": ["dsl"]})
    if internet:
        e["InternetService"] = internet
    contract = choice(t, {"Month-to-month": ["month to month", "monthly contract", "monthly plan"],
                          "Two year": ["two year", "2 year", "two years"], "One year": ["one year", "1 year"]})
    if contract:
        e["Contract"] = contract
    for field, phrases in (("TechSupport", ["tech support", "technical support"]),
                           ("OnlineSecurity", ["online security"])):
        v = flag(t, phrases)
        if v is not None:
            e[field] = "Yes" if v else "No"
    return e or None


META = {
    "name": "churn",
    "description": "Predicts whether a telecom customer will leave the company (churn)",
    "keywords": ["churn", "customer churn", "customer leave", "customer leaving", "switch provider",
                 "switch carrier", "switch companies", "cancel service", "telecom customer"],
    "example_input": {"tenure": 2, "InternetService": "Fiber optic", "Contract": "Month-to-month"},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/churn.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    specs = {f: ("num", 0, 120) if f == "tenure" else ("cat", b["categories"][f]) for f in b["features"]}
    row, imputed = build_row(payload, specs, b["typical_values"])
    pipeline = b["pipeline"]
    proba = pipeline.predict_proba(frame(row, specs))[0]
    risk = float(proba[list(pipeline.classes_).index(b["positive"])])
    other = next(c for c in b["classes"] if c != b["positive"])
    label = b["positive"] if risk >= 0.5 else other
    return {
        "prediction": f"likely to {label} (risk {round(risk * 100)} %)",
        "label": label,
        "risk_score": round(risk, 2),
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "Telco data from a fictional company (IBM). Academic reference.",
    }
