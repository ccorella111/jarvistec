from typing import Any, Optional

from pydantic import BaseModel


class CommandIn(BaseModel):
    text: str
    input: Optional[dict[str, Any]] = None  # if missing, the model's example input is used


class CommandOut(BaseModel):
    text: str
    model: Optional[str]
    default_input_used: bool = False
    result: Optional[dict[str, Any]] = None
    message: str
