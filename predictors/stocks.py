"""'stocks' model: predicts tomorrow's closing price of an S&P 500 stock (USD).

Trained in notebooks/stocks_sp500.ipynb (one global model for the 505 stocks).
Artifact: models/stocks.joblib (sklearn pipeline + last 60 closing prices of each stock).

Input (all optional):
    {"ticker": "AAPL"}                    -> uses the stored history of that stock (default AAPL)
    {"recent_closes": [..31 or more..]}   -> daily closing prices in USD, most recent last
Voice command: "JarvisTEC, price of Apple stock"  (the ticker is taken from the company name)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictors._common import has, normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stocks.joblib"

# Spoken company name -> ticker (all of them exist in the dataset)
ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "facebook": "FB", "netflix": "NFLX", "coca cola": "KO", "disney": "DIS", "walmart": "WMT",
    "exxon": "XOM", "jpmorgan": "JPM", "nike": "NKE", "pepsi": "PEP", "ibm": "IBM", "intel": "INTC",
    "boeing": "BA", "visa": "V", "mcdonalds": "MCD", "nvidia": "NVDA", "cisco": "CSCO",
    "oracle": "ORCL", "adobe": "ADBE", "paypal": "PYPL", "starbucks": "SBUX", "general electric": "GE",
    "verizon": "VZ", "goldman sachs": "GS", "bank of america": "BAC", "wells fargo": "WFC",
    "chevron": "CVX", "pfizer": "PFE", "johnson and johnson": "JNJ", "home depot": "HD",
    "mastercard": "MA", "salesforce": "CRM", "qualcomm": "QCOM", "costco": "COST", "micron": "MU",
    "caterpillar": "CAT", "ebay": "EBAY", "ford": "F",
}


def extract_input(text: str) -> dict | None:
    """Used by /command: looks for a company name in the phrase."""
    t = normalize(text)
    for name, ticker in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if has(t, [name]):
            return {"ticker": ticker}
    return None


META = {
    "name": "stocks",
    "description": "Predicts tomorrow's closing price of an S&P 500 stock (USD)",
    "keywords": ["stocks", "stock", "stock price", "share price", "stock market", "s&p 500", "wall street"]
                + [k for k in ALIASES if len(k) >= 4],
    "example_input": {"ticker": "AAPL"},
    "extract_input": extract_input,
}

FEATURES = ["ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_m7", "ret_m14", "ret_21",
            "vol_7", "vol_14", "vol_30", "p_ma7", "p_ma30"]
MIN_HISTORY = 31


def build_features(closes: pd.Series) -> pd.DataFrame:
    """Same function in training (notebook) and production: avoids train/serve skew."""
    c = closes.astype(float)
    r = np.log(c).diff()                              # daily log return
    X = pd.DataFrame(index=c.index)
    for i in range(1, 6):
        X[f"ret_{i}"] = r.shift(i - 1)                # returns of the last 5 days
    X["ret_m7"] = r.rolling(7).mean()
    X["ret_m14"] = r.rolling(14).mean()
    X["ret_21"] = np.log(c / c.shift(21))             # ~1-month momentum
    X["vol_7"] = r.rolling(7).std()                   # recent volatility
    X["vol_14"] = r.rolling(14).std()
    X["vol_30"] = r.rolling(30).std()
    X["p_ma7"] = np.log(c / c.rolling(7).mean())      # distance to the moving averages
    X["p_ma30"] = np.log(c / c.rolling(30).mean())
    return X[FEATURES]


_bundle = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} does not exist. Run notebooks/stocks_sp500.ipynb")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(payload: dict) -> dict:
    b = _load()
    payload = payload or {}
    ticker = str(payload.get("ticker") or "").upper().strip()
    closes = payload.get("recent_closes")

    if closes:
        try:
            series = pd.Series([float(x) for x in closes])
        except (TypeError, ValueError):
            raise ValueError("'recent_closes' must be a list of numbers")
        if len(series) < MIN_HISTORY or (series <= 0).any():
            raise ValueError(f"At least {MIN_HISTORY} positive daily closing prices are required")
        base_date, ticker = None, ticker or "CUSTOM"
    else:
        ticker = ticker or "AAPL"
        h = b["history"].get(ticker)
        if h is None:
            raise ValueError(f"'{ticker}' is not in the S&P 500 dataset. Examples: AAPL, MSFT, AMZN, KO, DIS")
        if len(h["closes"]) < MIN_HISTORY:
            raise ValueError(f"'{ticker}' does not have enough history ({len(h['closes'])} days)")
        series, base_date = pd.Series(h["closes"], dtype=float), h["last_date"]

    X = build_features(series).iloc[[-1]]
    if X.isna().any().any():
        raise ValueError("Not enough history to compute the model features")

    log_return = float(b["pipeline"].predict(X)[0])
    current = float(series.iloc[-1])
    prediction = current * float(np.exp(log_return))
    out = {
        "ticker": ticker,
        "prediction": round(prediction, 2),
        "unit": "USD",
        "current_price": round(current, 2),
        "change_pct": round((prediction / current - 1) * 100, 3),
        "model": b["model"],
        "note": "Academic reference; not financial advice.",
    }
    if base_date:
        out["base_date"] = base_date
        out["prediction_date"] = str((pd.Timestamp(base_date) + pd.offsets.BDay(1)).date())  # next business day
    return out
