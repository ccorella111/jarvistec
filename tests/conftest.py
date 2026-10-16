import os

# Tests must never call real services (this runs before the app is imported).
os.environ["STT_PROVIDER"] = "mock"
os.environ["EMOTION_PROVIDER"] = "mock"
