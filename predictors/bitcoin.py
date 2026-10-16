"""'bitcoin' model: predicts tomorrow's closing price of bitcoin (USD).

Trained in notebooks/bitcoin.ipynb. Artifact: models/bitcoin.joblib (dict with the sklearn
pipeline and the last 60 known closing prices).

Input (all optional):
    {}                                    -> uses the stored history (last data point: see 'base_date')
    {"recent_closes": [..30 or more..]}   -> daily closing prices in USD, most recent last
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bitcoin.joblib"

META = {
    "name": "bitcoin",
    "description": "Predicts tomorrow's closing price of bitcoin (USD)",
    "keywords": ["bitcoin price", "bitcoin", "cryptocurrency", "crypto", "btc"],
    "example_input": {},
}

# Features depend ONLY on the closing-price series, so the model can be used with new data.
FEATURES = ["ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_m7", "ret_m14",
            "vol_7", "vol_14", "p_ma7", "p_ma30"]
MIN_HISTORY = 30


def build_features(closes: pd.Series) -> pd.DataFrame:
    """Same function in training (notebook) and production: avoids train/serve skew."""
    c = closes.astype(float)
    r = np.log(c).diff()                       # daily log return
    X = pd.DataFrame(index=c.index)
    for i in range(1, 6):
        X[f"ret_{i}"] = r.shift(i - 1)         # returns of the last 5 days
    X["ret_m7"] = r.rolling(7).mean()
    X["ret_m14"] = r.rolling(14).mean()
    X["vol_7"] = r.rolling(7).std()            # recent volatility
    X["vol_14"] = r.rolling(14).std()
    X["p_ma7"] = np.log(c / c.rolling(7).mean())    # distance to the moving averages
    X["p_ma30"] = np.log(c / c.rolling(30).mean())
    return X[FEATURES]


_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/bitcoin.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    closes = (payload or {}).get("recent_closes")

    if closes:
        try:
            series = pd.Series([float(x) for x in closes])
        except (TypeError, ValueError):
            raise ValueError("'recent_closes' must be a list of numbers")
        if len(series) < MIN_HISTORY or (series <= 0).any():
            raise ValueError(f"At least {MIN_HISTORY} positive daily closing prices are required")
        base_date = None
    else:
        series = pd.Series(b["history"]["closes"], dtype=float)
        base_date = b["history"]["dates"][-1]

    X = build_features(series).iloc[[-1]]
    if X.isna().any().any():
        raise ValueError("Not enough history to compute the model features")

    log_return = float(b["pipeline"].predict(X)[0])
    current = float(series.iloc[-1])
    prediction = current * float(np.exp(log_return))

    out = {
        "prediction": round(prediction, 2),
        "unit": "USD",
        "current_price": round(current, 2),
        "change_pct": round((prediction / current - 1) * 100, 2),
        "model": b["model"],
        "note": "Academic reference; not financial advice.",
    }
    if base_date:
        out["base_date"] = base_date
        out["prediction_date"] = str((pd.Timestamp(base_date) + pd.Timedelta(days=1)).date())
    return out
