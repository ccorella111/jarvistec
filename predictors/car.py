"""'car' model: estimates the selling price of a used car (lakhs of rupees).

Trained in notebooks/car.ipynb (CarDekho data, India). Artifact: models/car.joblib.

Input (every field is optional; missing ones are filled with typical values):
    {"Year": 2015, "Present_Price": 8.0, "Kms_Driven": 40000, "Owner": 0,
     "Fuel_Type": "Petrol|Diesel|CNG", "Seller_Type": "Dealer|Individual", "Transmission": "Manual|Automatic"}
Voice command: "JarvisTEC, price of a 2015 diesel car with 40000 km"
"""
import re
from pathlib import Path

import joblib

from predictors._common import build_row, choice, frame, normalize, number

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "car.joblib"

CATEGORICAL = ["Fuel_Type", "Seller_Type", "Transmission"]
RANGES = {"Year": (1990, 2030), "Present_Price": (0.1, 100), "Kms_Driven": (0, 1_000_000), "Owner": (0, 5)}


def extract_input(text: str) -> dict | None:
    """Used by /command: reads year, kilometers, price and car type from the spoken phrase."""
    t = normalize(text)
    e: dict = {}
    m = re.search(r" (19[89]\d|20[0-3]\d) ", t)
    if m:
        e["Year"] = float(m.group(1))
    kms = number(t, after=["driven", "kms driven", "mileage"], before=["km", "kms", "kilometers", "kilometres"])
    if kms is not None:
        e["Kms_Driven"] = kms
    price = number(t, after=["present price", "showroom price", "new price", "ex showroom price", "new car price"])
    if price is not None:
        e["Present_Price"] = price
    for field, options in (("Fuel_Type", {"Petrol": ["petrol", "gasoline"], "Diesel": ["diesel"], "CNG": ["cng"]}),
                           ("Transmission", {"Automatic": ["automatic"], "Manual": ["manual"]}),
                           ("Seller_Type", {"Dealer": ["dealer", "dealership"], "Individual": ["individual", "private seller"]})):
        v = choice(t, options)
        if v:
            e[field] = v
    return e or None


META = {
    "name": "car",
    "description": "Estimates the selling price of a used car (lakhs of rupees)",
    "keywords": ["car price", "price of a car", "price of a used car", "used car", "car", "vehicle", "automobile"],
    "example_input": {"Year": 2015, "Present_Price": 8.0, "Kms_Driven": 40000},
    "extract_input": extract_input,
}

_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/car.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    specs = {f: ("cat", b["categories"][f]) if f in CATEGORICAL else ("num", *RANGES[f]) for f in b["features"]}
    row, imputed = build_row(payload, specs, b["typical_values"])
    price = float(b["pipeline"].predict(frame(row, specs))[0])
    return {
        "prediction": round(price, 2),
        "unit": "lakhs of rupees",
        "inr_equivalent": int(round(price * 100_000)),
        "imputed_fields": imputed,
        "model": b["model"],
        "note": "1 lakh = 100,000 rupees. CarDekho data (India), cars from 2003 to 2018.",
    }
