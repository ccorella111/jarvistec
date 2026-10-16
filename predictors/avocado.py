"""'avocado' model: predicts the average price of a Hass avocado next week (USD).

Trained in notebooks/avocado.ipynb (Hass Avocado Board data). One global model for all
region/type series. Artifact: models/avocado.joblib (pipeline + last 16 weekly prices of each series).

Input (all optional):
    {"region": "Boston", "type": "organic|conventional"}      -> defaults: TotalUS, conventional
    {"recent_prices": [..13 or more weekly prices..], "last_date": "2018-03-25"}   -> your own series
Voice command: "JarvisTEC, price of organic avocados in Boston next week"
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictors._common import has, normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "avocado.joblib"

FEATURES = ["price", "change_1", "change_2", "dist_mean_4", "dist_mean_12", "sin_week", "cos_week"]
MIN_HISTORY = 13


def build_features(prices: pd.Series) -> pd.DataFrame:
    """Same function in training (notebook) and production. `prices` is a weekly series indexed by date."""
    s = prices.astype(float)
    angle = 2 * np.pi * s.index.dayofyear / 365.25
    return pd.DataFrame({
        "price": s,
        "change_1": s - s.shift(1),                    # change in the last week
        "change_2": s.shift(1) - s.shift(2),           # change the week before
        "dist_mean_4": s - s.rolling(4).mean(),        # distance to the last month's mean
        "dist_mean_12": s - s.rolling(12).mean(),      # distance to the last three months' mean
        "sin_week": np.sin(angle), "cos_week": np.cos(angle),   # time of year
    }, index=s.index)[FEATURES]


_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/avocado.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def extract_input(text: str) -> dict | None:
    """Used by /command: looks for a region name and the avocado type in the phrase."""
    t = normalize(text)
    e: dict = {}
    if has(t, ["organic"]):
        e["type"] = "organic"
    elif has(t, ["conventional"]):
        e["type"] = "conventional"
    if has(t, ["national", "nationwide", "united states", "us total", "whole country"]):
        e["region"] = "TotalUS"
    else:
        try:
            regions = _load()["regions"]
        except FileNotFoundError:
            regions = []
        squashed = t.replace(" ", "")
        found = [r for r in regions if re.sub(r"[^a-z0-9]", "", r.lower()) in squashed]
        if found:
            e["region"] = max(found, key=len)           # the longest match wins ("SanDiego" over "Diego")
    return e or None


META = {
    "name": "avocado",
    "description": "Predicts next week's average price of a Hass avocado (USD) for a US region and type",
    "keywords": ["avocado", "avocados", "avocado price", "price of avocados", "guacamole"],
    "example_input": {"region": "Boston", "type": "organic"},
    "extract_input": extract_input,
}


def predict(payload: dict) -> dict:
    b = _load()
    payload = payload or {}
    extra = set(payload) - {"region", "type", "recent_prices", "last_date"}
    if extra:
        raise ValueError(f"Unknown fields: {sorted(extra)}. Valid: region, type, recent_prices, last_date")
    region = str(payload.get("region") or "TotalUS").strip()
    kind = str(payload.get("type") or "conventional").strip().lower()
    if kind not in ("conventional", "organic"):
        raise ValueError("'type' must be 'conventional' or 'organic'")

    if payload.get("recent_prices"):
        try:
            values = [float(x) for x in payload["recent_prices"]]
            last = pd.Timestamp(payload["last_date"])
        except (TypeError, ValueError, KeyError):
            raise ValueError("'recent_prices' must be a list of numbers and 'last_date' a date (YYYY-MM-DD)")
        if len(values) < MIN_HISTORY or min(values) <= 0:
            raise ValueError(f"At least {MIN_HISTORY} positive weekly prices are required")
        dates = pd.date_range(end=last, periods=len(values), freq="7D")
        series, region = pd.Series(values, index=dates), region if payload.get("region") else "CUSTOM"
    else:
        regions = {r.lower(): r for r in b["regions"]}
        if region.lower() not in regions:
            raise ValueError(f"Unknown region '{region}'. Examples: TotalUS, Boston, California, West")
        region = regions[region.lower()]
        h = b["history"].get(f"{region}|{kind}")
        if h is None:
            raise ValueError(f"There is no history for {region} ({kind})")
        series = pd.Series(h["prices"], index=pd.to_datetime(h["dates"]))

    X = build_features(series).iloc[[-1]]
    if X.isna().any().any():
        raise ValueError("Not enough history to compute the model features")
    current = float(series.iloc[-1])
    prediction = current + float(b["pipeline"].predict(X)[0])      # the model predicts the weekly CHANGE
    base = series.index[-1]
    return {
        "prediction": round(prediction, 2),
        "unit": "USD per avocado",
        "current_price": round(current, 2),
        "change_pct": round((prediction / current - 1) * 100, 1),
        "region": region,
        "type": kind,
        "base_date": str(base.date()),
        "prediction_date": str((base + pd.Timedelta(days=7)).date()),
        "model": b["model"],
        "note": "Weekly average price in the United States (Hass Avocado Board). Academic reference.",
    }
