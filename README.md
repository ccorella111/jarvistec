# JarvisTEC

Personal assistant with emotion recognition (Google Cloud Vision), speech to text (Azure Speech) and 10 machine learning models exposed through a REST API.

## Structure
```
app/          FastAPI API (routers, Azure/Google services, voice-command interpretation)
predictors/   One file per ML model: META + predict()  (registered automatically)
models/       Trained models (.joblib)   [committed: they are small]
data/         Datasets                   [not committed]
notebooks/    One notebook per model (rubric template: the 6 stages)
scripts/      train_all.py - re-executes the notebooks and regenerates models/
frontend/     Web interface: index.html (structure + JS) and style.css (styles)
docs/         Document sources / figures
tests/        API tests
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # fill in the keys (see the comments in the file)
uvicorn app.main:app --reload
```
- JarvisTEC interface: http://localhost:8000/ui/  (use Chrome or Edge)
- Interactive API docs: http://localhost:8000/docs
- No cloud keys yet? Set `STT_PROVIDER=mock` and `EMOTION_PROVIDER=mock` in `.env`.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health: status, models loaded, voice/emotion providers |
| GET | `/models` | Model list with description and example input |
| POST | `/predict/{name}` | Run one model with a JSON input (missing fields use typical values) |
| POST | `/command` | Text command, e.g. `{"text": "JarvisTEC, bitcoin price"}` |
| POST | `/voice` | Speech to text (multipart, field `audio`, WAV 16 kHz) |
| POST | `/voice/command` | Audio -> text -> model result |
| POST | `/emotion` | Dominant emotion from a photo (multipart, field `image`) |

## How to add a model
1. Train it in `notebooks/<name>.ipynb` (see `notebooks/bitcoin.ipynb` for a complete example, `notebooks/template_model.ipynb` for the empty template).
2. Save the pipeline: `joblib.dump(bundle, "models/<name>.joblib")`.
3. Copy `predictors/_template.py` to `predictors/<name>.py` and complete it.
4. Done: it shows up in `GET /models`, `POST /predict/<name>` and in the voice commands.

## Team split
| Person | Role | Models |
|---|---|---|
| A | Platform (API, interface, voice, emotion) | - |
| B | Models | 1-5 |
| C | Models | 6-10 |

## Git flow
`main` (stable) <- `dev` <- `feature/<name>`. Everyone works on their own branch and opens a PR to `dev`.

## Models (10 of 10)
| Model | Notebook | Predictor | Dataset (in `data/`) | Example voice command |
|---|---|---|---|---|
| bitcoin (tomorrow's close) | `notebooks/bitcoin.ipynb` | `predictors/bitcoin.py` | `bitcoin_price_*.csv` | "JarvisTEC, bitcoin price" |
| stocks S&P 500 (tomorrow's close) | `notebooks/stocks_sp500.ipynb` | `predictors/stocks.py` | `all_stocks_5yr.csv` (29 MB, not in the zip) | "JarvisTEC, stock price of Apple" |
| wine (low/medium/high quality) | `notebooks/wine.ipynb` | `predictors/wine.py` | `winequalityN.csv` | "JarvisTEC, quality of a red wine with alcohol 12" |
| stroke (stroke risk) | `notebooks/stroke.ipynb` | `predictors/stroke.py` | `healthcare-dataset-stroke-data.csv` | "JarvisTEC, stroke risk of a 70 year old man with hypertension" |
| car (used car price) | `notebooks/car.ipynb` | `predictors/car.py` | `car_data.csv` (downloaded if missing) | "JarvisTEC, price of a used car" |
| churn (telecom customer leaving) | `notebooks/churn.ipynb` | `predictors/churn.py` | Telco churn CSV (downloaded if missing) | "JarvisTEC, will this customer churn" |
| cirrhosis (disease stage) | `notebooks/cirrhosis.ipynb` | `predictors/cirrhosis.py` | `cirrhosis.csv` | "JarvisTEC, cirrhosis stage" |
| body_fat (body fat %) | `notebooks/body_fat.ipynb` | `predictors/body_fat.py` | `bodyfat.csv` | "JarvisTEC, body fat percentage" |
| hepatitis (type of hepatitis) | `notebooks/hepatitis.ipynb` | `predictors/hepatitis.py` | `HepatitisCdata.csv` | "JarvisTEC, hepatitis type" |
| avocado (next week's price) | `notebooks/avocado.ipynb` | `predictors/avocado.py` | `avocado.csv` | "JarvisTEC, avocado price" |

Retrain everything (or only some): `python scripts/train_all.py [name ...]`.

If a `.joblib` grows past ~50 MB, save it with `compress=3` or use Git LFS (GitHub rejects files over 100 MB).

Datasets (`data/*.csv`) are not committed: each member downloads them from Kaggle and puts them in `data/`
(e.g. `data/all_stocks_5yr.csv`) only when re-running a notebook.

## Integrating into an existing project (what to copy)
| Copy | Why |
|---|---|
| `app/` and `predictors/` | The whole API and the 10 models (entire folders) |
| `models/*.joblib` | The trained models (predictors do not work without them) |
| `frontend/index.html` + `frontend/style.css` | The interface (both files together, in the same folder) |
| `notebooks/` | The 10 executed notebooks + `template_model.ipynb` |
| `scripts/train_all.py` | One-command retraining |
| `requirements.txt`, `.env.example`, `.gitignore` | Dependencies (scikit-learn pinned to the training version) and configuration |
| `data/` | Only if you want to re-run the notebooks |
| `tests/` | Tests (`pytest`) |

The notebooks import `predictors/` with `sys.path.insert(0, '..')`, so they must stay in `notebooks/`, next to `predictors/`, and be run from that folder.
