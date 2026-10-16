"""API tests. Model tests need the trained models/*.joblib files; they are skipped if a model is missing."""
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODELS = Path(__file__).resolve().parent.parent / "models"


def needs(name):
    return pytest.mark.skipif(not (MODELS / f"{name}.joblib").exists(), reason=f"models/{name}.joblib is missing")


def test_health_and_frontend():
    with TestClient(app) as c:
        h = c.get("/").json()
        assert h["status"] == "ok" and h["voice"] == "mock" and h["emotion"] == "mock"
        html = c.get("/ui/").text
        assert '<link rel="stylesheet" href="style.css">' in html and "<style" not in html
        css = c.get("/ui/style.css")
        assert css.status_code == 200 and "text/css" in css.headers["content-type"] and "--cy" in css.text


def test_all_models_registered():
    with TestClient(app) as c:
        names = {m["name"] for m in c.get("/models").json()}
        assert names == {"bitcoin", "stocks", "stroke", "wine", "car", "churn", "cirrhosis",
                         "body_fat", "hepatitis", "avocado"}
        assert c.post("/predict/nope", json={}).status_code == 404


def test_mock_emotion_and_voice():
    with TestClient(app) as c:
        assert c.post("/emotion", files={"image": ("x.jpg", b"123", "image/jpeg")}).json()["emotion"]
        r = c.post("/voice/command", files={"audio": ("a.wav", b"123", "audio/wav")}).json()
        assert r["transcription"]["text"] and r["command"]["model"] == "wine"


def test_unknown_command():
    with TestClient(app) as c:
        r = c.post("/command", json={"text": "tell me a joke"}).json()
        assert r["model"] is None and "did not understand" in r["message"]


@needs("bitcoin")
def test_bitcoin():
    with TestClient(app) as c:
        r = c.post("/predict/bitcoin", json={}).json()
        assert r["unit"] == "USD" and math.isfinite(r["prediction"]) and r["prediction"] > 0 and r["base_date"]
        closes = [3000 + i * 5 for i in range(40)]
        r2 = c.post("/predict/bitcoin", json={"recent_closes": closes}).json()
        assert abs(r2["current_price"] - closes[-1]) < 0.01 and "base_date" not in r2
        assert c.post("/predict/bitcoin", json={"recent_closes": [1, 2, 3]}).status_code == 422
        cmd = c.post("/command", json={"text": "JarvisTEC, what is the bitcoin price for tomorrow?"}).json()
        assert cmd["model"] == "bitcoin" and cmd["result"]["unit"] == "USD"


@needs("stocks")
def test_stocks():
    with TestClient(app) as c:
        r = c.post("/predict/stocks", json={"ticker": "aapl"}).json()
        assert r["ticker"] == "AAPL" and r["prediction"] > 0 and r["prediction_date"]
        assert c.post("/predict/stocks", json={}).json()["ticker"] == "AAPL"
        assert c.post("/predict/stocks", json={"ticker": "ZZZZ"}).status_code == 422
        assert c.post("/predict/stocks", json={"recent_closes": [1, 2]}).status_code == 422
        r = c.post("/command", json={"text": "JarvisTEC, price of Microsoft stock"}).json()
        assert r["model"] == "stocks" and r["result"]["ticker"] == "MSFT" and not r["default_input_used"]
        r = c.post("/command", json={"text": "JarvisTEC, how much is Coca-Cola on the stock market?"}).json()
        assert r["result"]["ticker"] == "KO"


@needs("stroke")
def test_stroke():
    high = {"gender": "Male", "age": 78, "hypertension": 1, "heart_disease": 1, "ever_married": "Yes",
            "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 220, "bmi": 31,
            "smoking_status": "formerly smoked"}
    low = {"gender": "Female", "age": 28, "hypertension": 0, "heart_disease": 0, "ever_married": "No",
           "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 85, "bmi": 22,
           "smoking_status": "never smoked"}
    with TestClient(app) as c:
        a, b = c.post("/predict/stroke", json=high).json(), c.post("/predict/stroke", json=low).json()
        assert a["alert"] is True and a["risk_level"] == "high" and b["risk_level"] == "low"
        assert 0 <= b["probability"] < a["probability"] <= 1
        assert "bmi" in c.post("/predict/stroke", json={"age": 70}).json()["imputed_fields"]
        assert c.post("/predict/stroke", json={"gender": "Male"}).status_code == 422       # age is missing
        assert c.post("/predict/stroke", json={"age": 200}).status_code == 422
        assert c.post("/predict/stroke", json={"age": 50, "gender": "Robot"}).status_code == 422
        r = c.post("/command", json={"text": "JarvisTEC, stroke risk of a 70 year old man with hypertension"}).json()
        assert r["model"] == "stroke" and not r["default_input_used"] and "risk" in r["result"]["prediction"]
        r = c.post("/command", json={"text": "JarvisTEC, stroke risk with hypertension"}).json()
        assert r["model"] == "stroke" and "age" in r["message"]                             # asks for the age


@needs("wine")
def test_wine():
    high = {"type": "white", "alcohol": 12.8, "volatile_acidity": 0.2, "density": 0.991, "chlorides": 0.03}
    low = {"type": "red", "alcohol": 9.0, "volatile_acidity": 0.85, "chlorides": 0.11, "density": 0.9985}
    with TestClient(app) as c:
        a, b = c.post("/predict/wine", json=high).json(), c.post("/predict/wine", json=low).json()
        assert a["class"] == "high" and b["class"] == "low"
        assert abs(sum(a["probabilities"].values()) - 1) < 1e-3 and "pH" in a["imputed_fields"]
        assert c.post("/predict/wine", json={}).status_code == 422
        assert c.post("/predict/wine", json={"alcohol": 40}).status_code == 422
        assert c.post("/predict/wine", json={"type": "rose", "alcohol": 11}).status_code == 422
        r = c.post("/command", json={"text": "JarvisTEC, quality of a red wine with alcohol 12 5 and pH 3 3"}).json()
        assert r["model"] == "wine" and not r["default_input_used"]
        assert "alcohol" not in r["result"]["imputed_fields"] and "pH" not in r["result"]["imputed_fields"]


@needs("car")
def test_car():
    with TestClient(app) as c:
        r = c.post("/predict/car", json={"Year": 2015, "Present_Price": 8.0, "Kms_Driven": 40000}).json()
        assert r["unit"] == "lakhs of rupees" and r["prediction"] > 0
        assert c.post("/predict/car", json={"Fuel_Type": "Steam"}).status_code == 422
        r = c.post("/command", json={"text": "JarvisTEC, price of a 2015 diesel car with 40000 km"}).json()
        assert r["model"] == "car" and r["result"]["prediction"] > 0 and "Fuel_Type" not in r["result"]["imputed_fields"]


@needs("churn")
def test_churn():
    with TestClient(app) as c:
        new = c.post("/predict/churn", json={"tenure": 2, "InternetService": "Fiber optic", "Contract": "Month-to-month"}).json()
        loyal = c.post("/predict/churn", json={"tenure": 60, "InternetService": "DSL", "Contract": "Two year"}).json()
        assert new["risk_score"] > loyal["risk_score"] and new["label"] == "churn" and loyal["label"] == "stay"
        r = c.post("/command", json={"text": "JarvisTEC, will a customer with 2 months on a month to month fiber optic plan churn?"}).json()
        assert r["model"] == "churn" and r["result"]["label"] == "churn"


@needs("cirrhosis")
def test_cirrhosis():
    with TestClient(app) as c:
        r = c.post("/predict/cirrhosis", json={}).json()
        assert r["prediction"].startswith("stage") and abs(sum(r["probabilities"].values()) - 1) < 0.05
        assert c.post("/predict/cirrhosis", json={"Sex": "X"}).status_code == 422
        assert c.post("/command", json={"text": "JarvisTEC, which cirrhosis stage is a 60 year old woman with ascites?"}).json()["model"] == "cirrhosis"


@needs("body_fat")
def test_body_fat():
    with TestClient(app) as c:
        lean = c.post("/predict/body_fat", json={"Age": 25, "Height": 180, "Abdomen": 78, "Wrist": 18}).json()
        heavy = c.post("/predict/body_fat", json={"Age": 55, "Height": 172, "Abdomen": 112, "Wrist": 18.5}).json()
        assert lean["prediction"] < heavy["prediction"] and lean["unit"] == "% body fat"
        assert c.post("/predict/body_fat", json={"Abdomen": 10}).status_code == 422
        assert c.post("/command", json={"text": "JarvisTEC, body fat of a man with abdomen 95"}).json()["model"] == "body_fat"


@needs("hepatitis")
def test_hepatitis():
    with TestClient(app) as c:
        r = c.post("/predict/hepatitis", json={"Age": 56, "Sex": "m", "ALB": 33.0, "AST": 92.9, "BIL": 34.0, "CHE": 3.4}).json()
        assert r["prediction"] in {"healthy", "hepatitis", "fibrosis", "cirrhosis"}
        assert c.post("/predict/hepatitis", json={"Sex": "x"}).status_code == 422
        assert c.post("/command", json={"text": "JarvisTEC, which type of hepatitis does a 56 year old man with AST 92 have?"}).json()["model"] == "hepatitis"


@needs("avocado")
def test_avocado():
    with TestClient(app) as c:
        r = c.post("/predict/avocado", json={"region": "Boston", "type": "organic"}).json()
        assert r["unit"] == "USD per avocado" and r["prediction"] > 0 and r["region"] == "Boston"
        assert c.post("/predict/avocado", json={}).json()["region"] == "TotalUS"
        assert c.post("/predict/avocado", json={"region": "Atlantis"}).status_code == 422
        r = c.post("/command", json={"text": "JarvisTEC, price of organic avocados in Boston next week"}).json()
        assert r["model"] == "avocado" and r["result"]["region"] == "Boston" and r["result"]["type"] == "organic"
