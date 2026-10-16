from typing import Any

from fastapi import APIRouter, HTTPException

from app.registry import REGISTRY

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models():
    return [
        {"name": m["name"], "description": m["description"], "keywords": m["keywords"],
         "example_input": m["example_input"]}
        for m in REGISTRY.values()
    ]


@router.post("/predict/{name}")
def predict(name: str, payload: dict[str, Any]):
    if name not in REGISTRY:
        raise HTTPException(404, f"Model '{name}' does not exist. See GET /models")
    try:
        return REGISTRY[name]["function"](payload)
    except ValueError as e:
        raise HTTPException(422, str(e))
