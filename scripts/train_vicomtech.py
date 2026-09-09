"""Train the turning-native regressor on Vicomtech segments.

HistGradientBoosting (handles NaNs natively) on the 152 native
features, split by tool: train 9 tools / val 2 / test 2.
Saves models/vicomtech_hgbr.joblib.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/train_vicomtech.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np

from hybrid_ai.ann.vicomtech_model import VicomtechConfig, VicomtechTurningModel
from hybrid_ai.core.vicomtech_loader import load_vicomtech


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    t0 = time.time()
    Xtr, ytr, _ = load_vicomtech(split="train")
    Xva, yva, _ = load_vicomtech(split="val")
    Xte, yte, gte = load_vicomtech(split="test")
    print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape} "
          f"tools={sorted(set(gte.tolist()))}")

    base = rmse(yte, ytr.mean(axis=0))
    print(f"mean-baseline test RMSE: {base:.4f}")
    model = VicomtechTurningModel(config=VicomtechConfig())
    model.fit(np.vstack([Xtr, Xva]), np.vstack([ytr, yva]))
    out = ROOT / "models" / "vicomtech_hgbr.joblib"
    model.save(out)
    print(f"saved {out} in {time.time() - t0:.0f}s")
    pred = model.predict(Xte)
    print(f"test RMSE {rmse(yte, pred):.4f} "
          f"(VB {rmse(yte[:, 0], pred[:, 0]):.4f}, Ra {rmse(yte[:, 1], pred[:, 1]):.4f})")


if __name__ == "__main__":
    main()
