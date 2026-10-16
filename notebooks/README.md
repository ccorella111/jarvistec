# Notebooks

Every notebook follows the rubric template (problem analysis, data understanding and exploration,
model, evaluation, conclusion) and ends by exporting `models/<name>.joblib`. Run them **from this folder**
(or all at once with `python scripts/train_all.py` from the repo root).

| Notebook | Task | Main result (test set) | Exports |
|---|---|---|---|
| `bitcoin.ipynb` | Regression: bitcoin's next close | Ties the "tomorrow = today" baseline | `models/bitcoin.joblib` |
| `stocks_sp500.ipynb` | Regression: next close of 505 stocks | Ties the baseline; error about 1 % (MAPE) | `models/stocks.joblib` |
| `stroke.ipynb` | Classification: stroke risk (imbalanced classes) | ROC-AUC 0.85, recall 76 % | `models/stroke.joblib` |
| `wine.ipynb` | Classification: low/medium/high quality | Accuracy 63 %, macro F1 0.61 | `models/wine.joblib` |
| `car.ipynb` | Regression: used car price | CV R² 0.94 | `models/car.joblib` |
| `churn.ipynb` | Classification: telecom customer churn | Recall 0.78, ROC-AUC 0.83 | `models/churn.joblib` |
| `cirrhosis.ipynb` | Classification: liver disease stage (1-2 / 3 / 4) | Accuracy 53 % (baseline 38 %), weighted kappa 0.43 | `models/cirrhosis.joblib` |
| `body_fat.ipynb` | Regression: body fat percentage | MAE 3.7 points, R² 0.69 (baseline 6.6) | `models/body_fat.joblib` |
| `hepatitis.ipynb` | Classification: healthy / hepatitis C / fibrosis / cirrhosis | Detects 96 % of sick patients (3 % false alarms); macro F1 0.75 (CV) | `models/hepatitis.joblib` |
| `avocado.ipynb` | Regression: next week's avocado price | RMSE 15 % below the naive baseline on test (ties on validation) | `models/avocado.joblib` |
| `template_model.ipynb` | Empty template for the next model | - | - |

Datasets go in `data/` (see the root `README.md`). `car.ipynb` and `churn.ipynb` download theirs if the file is missing.
