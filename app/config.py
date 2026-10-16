import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The Google credentials path is made absolute (relative to the repo root),
# so it works no matter where uvicorn is launched from.
_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
if _cred:
    p = Path(_cred)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p if p.is_absolute() else (ROOT / p).resolve())


class Settings:
    STT_PROVIDER = os.getenv("STT_PROVIDER", "mock").lower()          # azure | mock
    EMOTION_PROVIDER = os.getenv("EMOTION_PROVIDER", "mock").lower()  # google | mock
    AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
    SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-US")


settings = Settings()
