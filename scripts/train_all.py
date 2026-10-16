"""Executes every notebook in notebooks/ in place (also regenerating models/*.joblib).

Usage (from the repo root):    python scripts/train_all.py [name ...]
Examples:                      python scripts/train_all.py            # all notebooks
                               python scripts/train_all.py car churn   # only these

Each notebook needs its dataset in data/ (see notebooks/README.md). Needs: pip install -r requirements.txt
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ["bitcoin", "stocks_sp500", "stroke", "wine", "car", "churn", "cirrhosis", "body_fat", "hepatitis", "avocado"]


def main() -> int:
    wanted = sys.argv[1:] or NOTEBOOKS
    failed = []
    for name in wanted:
        nb = ROOT / "notebooks" / f"{name}.ipynb"
        if not nb.exists():
            nb = ROOT / "notebooks" / f"{name}_sp500.ipynb" if name == "stocks" else nb
        print(f"\n=== {nb.name}")
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", nb.name], cwd=nb.parent)
        if r.returncode:
            failed.append(nb.name)
    print("\nDone." if not failed else f"\nFailed: {failed} (is the dataset in data/?)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
