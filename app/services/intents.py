"""Command interpretation: spoken phrase -> model.
"""
from predictors._common import has, normalize

WAKE_WORDS = ("jarvistec", "jarvis tec", "jarvis")


def strip_wake_word(text: str) -> str:
    t = normalize(text).strip()
    for w in WAKE_WORDS:
        if t.startswith(w):
            return " " + t[len(w):].strip() + " "
    return " " + t + " "


def detect_model(text: str, registry: dict) -> str | None:
    """Returns the model with the longest matching keywords (whole-word matches only)."""
    t = strip_wake_word(text)
    best, best_score = None, 0
    for name, meta in registry.items():
        score = sum(len(normalize(k).strip()) for k in meta["keywords"] if has(t, [k]))
        if score > best_score:
            best, best_score = name, score
    return best
