"""TEMPLATE: copy to predictors/<name>.py (without the leading underscore) to register a new model.

Contract rules:
  1. The trained model (with its preprocessing, e.g. an sklearn Pipeline) is saved in models/<name>.joblib
  2. META describes the model and its voice-command keywords.
  3. predict(payload: dict) -> dict. It always returns a JSON-serializable dictionary.
     Raise ValueError for invalid input (the API turns it into HTTP 422).
"""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "NAME.joblib"

META = {
    "name": "NAME",
    "description": "What this model predicts",
    "keywords": ["phrase 1", "phrase 2"],     
    "example_input": {"column1": 0.0, "column2": 0.0},
}

_model = None


def predict(payload: dict) -> dict:
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    value = _model.predict(pd.DataFrame([payload]))[0]
    return {"prediction": float(value), "unit": "?"}
