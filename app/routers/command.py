from fastapi import APIRouter

from app.registry import REGISTRY
from app.schemas import CommandIn, CommandOut
from app.services.intents import detect_model

router = APIRouter(prefix="/command", tags=["commands"])


def run_command(text: str, payload: dict | None = None) -> CommandOut:
    name = detect_model(text, REGISTRY)
    if name is None:
        return CommandOut(text=text, model=None,
                          message="I did not understand the command. Try GET /models to see what I can do.")
    meta = REGISTRY[name]
    uses_default = payload is None
    if payload is None and meta.get("extract_input"):
        extracted = meta["extract_input"](text)
        if extracted:
            payload, uses_default = extracted, False
    try:
        result = meta["function"](payload if payload is not None else meta["example_input"])
    except ValueError as e:
        return CommandOut(text=text, model=name, message=f"Invalid input: {e}")
    return CommandOut(text=text, model=name, default_input_used=uses_default, result=result,
                      message=f"Ran the '{name}' model.")


@router.post("", response_model=CommandOut)
def command(body: CommandIn):
    """Interprets a text (e.g. 'JarvisTEC, bitcoin price') and runs the matching model."""
    return run_command(body.text, body.input)
