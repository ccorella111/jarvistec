"""Helpers shared by all predictors: text normalization, voice-command parsing and input validation."""
import re
import unicodedata

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- text / voice commands
def normalize(text: str) -> str:
    """Lowercase, no accents or punctuation, single spaces, padded with one space on each side."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip() + " "


def has(text: str, phrases) -> bool:
    """True if any phrase appears as whole words in the (already normalized) text."""
    return any(f" {normalize(p).strip()} " in text for p in phrases)


def _to_float(a: str, b: str | None) -> float:
    return float(f"{a}.{b}") if b else float(a)


_NUM = r"(\d{1,6})(?: (\d{1,3}))?"      # "12 5" is how the normalized text spells 12.5


def number(text: str, after=(), before=()) -> float | None:
    """A number spoken after a keyword ('bmi 28.5') or before a unit ('40000 km')."""
    for kw in after:
        k = normalize(kw).strip()
        m = re.search(rf" {k} (?:of |is |at |was |about |around |to )?{_NUM} ", text)
        if m:
            return _to_float(*m.groups())
    for unit in before:
        u = normalize(unit).strip()
        m = re.search(rf" {_NUM} {u} ", text)
        if m:
            return _to_float(*m.groups())
    return None


def flag(text: str, phrases) -> int | None:
    """1 if the condition is mentioned, 0 if it is negated ('no hypertension'), None if not mentioned."""
    present = False
    for p in phrases:
        p = normalize(p).strip()
        if re.search(rf" (?:no|without|not|never|free of|denies) (?:[a-z0-9]+ ){{0,3}}{p} ", text):
            return 0
        present = present or f" {p} " in text
    return 1 if present else None


def choice(text: str, options: dict):
    """First option (in dict order) whose phrases appear in the text."""
    for value, phrases in options.items():
        if has(text, phrases):
            return value
    return None


# ---------------------------------------------------------------- input validation
def check_number(field: str, value, low: float, high: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number")
    if not (low <= x <= high):
        raise ValueError(f"'{field}' is outside the reasonable range ({low}-{high})")
    return x


def check_binary(field: str, value) -> float:
    if isinstance(value, str):
        value = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}.get(value.strip().lower(), value)
    if value in (0, 1, True, False):
        return float(value)
    raise ValueError(f"'{field}' must be 0 or 1 (or yes/no)")


def check_category(field: str, value, options, aliases: dict | None = None) -> str:
    s = str(value).strip()
    for o in options:
        if s.lower() == str(o).lower():
            return o
    if aliases and aliases.get(s.lower()) in options:
        return aliases[s.lower()]
    raise ValueError(f"'{field}' must be one of: {', '.join(map(str, options))}")


def build_row(payload: dict, specs: dict, typical: dict | None = None) -> tuple[dict, list[str]]:
    """Validates the input and returns (row, imputed_fields).

    specs: {field: ("num", low, high) | ("bin",) | ("cat", options[, aliases])}
    Missing fields are filled with `typical` values (or NaN when typical is None).
    """
    payload = payload or {}
    extra = set(payload) - set(specs)
    if extra:
        raise ValueError(f"Unknown fields: {sorted(extra)}. Valid fields: {list(specs)}")
    row, imputed = {}, []
    for field, spec in specs.items():
        v = payload.get(field)
        if v is None or v == "":
            row[field] = typical[field] if typical is not None else np.nan
            imputed.append(field)
        elif spec[0] == "num":
            row[field] = check_number(field, v, spec[1], spec[2])
        elif spec[0] == "bin":
            row[field] = check_binary(field, v)
        else:
            row[field] = check_category(field, v, spec[1], spec[2] if len(spec) > 2 else None)
    return row, imputed


def frame(row: dict, specs: dict) -> pd.DataFrame:
    """One-row DataFrame with numeric columns as float."""
    df = pd.DataFrame([row], columns=list(specs))
    for field, spec in specs.items():
        if spec[0] in ("num", "bin"):
            df[field] = df[field].astype(float)
    return df
