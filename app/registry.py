"""Automatic model registry.

Every file in predictors/ whose name does not start with "_" is discovered automatically.
To add a model you do NOT touch the API: create predictors/<name>.py with META and predict()
(see predictors/_template.py).
"""
import importlib
import pkgutil

import predictors

REGISTRY: dict[str, dict] = {}


def load_models() -> None:
    REGISTRY.clear()
    for mod in pkgutil.iter_modules(predictors.__path__):
        if mod.name.startswith("_"):
            continue
        m = importlib.import_module(f"predictors.{mod.name}")
        meta = dict(m.META)
        meta["function"] = m.predict
        REGISTRY[meta["name"]] = meta
