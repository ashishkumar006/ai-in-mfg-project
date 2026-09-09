"""Test-set report for the 1M-param CNN-BiLSTM (models/phm_cnn_bilstm.pt).

Evaluates ONLY on held-out cutter c6 (2,152 windows the model never
trained on), prints full metrics, and saves plots to
demo_outputs/model_test/:
  - trajectory.png: true vs predicted wear over the cutter's life
  - scatter.png: predicted vs true VB with the ideal line
  - residual.png: error histogram + error-vs-wear

Usage (from the project root)::
    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/test_big_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_ai.ann.model import WearRoughnessDL
from hybrid_ai.core.phm_loader import load_phm_windows

OUT = ROOT / "demo_outputs" / "model_test"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = WearRoughnessDL.load(ROOT / "models" / "phm_cnn_bilstm.pt")
    n_params = sum(p.numel() for p in model.model.parameters())
    Xtr, ytr, gtr = load_phm_windows(split="train")
    Xte, yte, gte = load_phm_windows(split="test")
    assert set(gte.tolist()) == {"c6"} and not (set(gtr.tolist()) & set(gte.tolist()))
    print(f"model params: {n_params} | train cutters: {sorted(set(gtr.tolist()))} "
          f"({len(Xtr)} windows) | TEST cutter: c6 ({len(Xte)} windows)")

    t0 = time.time()
    pred = model.predict(Xte)
    print(f"predicted {len(Xte)} test windows in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / len(Xte) * 1000:.1f} ms each)")

    true_vb, pvb = yte[:, 0], pred[:, 0]
    base = float(np.sqrt(np.mean((yte - ytr.mean(axis=0)) ** 2)))
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    resid = true_vb - pvb
    print(f"joint RMSE {rmse:.4f} vs mean-baseline {base:.4f} "
          f"({base / rmse:.1f}x better)")
    print(f"VB: rmse={np.sqrt(np.mean(resid ** 2)):.4f} "
          f"mae={np.mean(np.abs(resid)):.4f} bias={resid.mean():+.4f} "
          f"max|err|={np.abs(resid).max():.4f}")
    within_20um = float(np.mean(np.abs(resid) <= 0.020)) * 100
    within_40um = float(np.mean(np.abs(resid) <= 0.040)) * 100
    print(f"within 20um: {within_20um:.1f}% | within 40um: {within_40um:.1f}%")

    order = np.argsort(true_vb)
    plt.figure(figsize=(10, 4))
    plt.plot(true_vb[order], label="true VB", linewidth=1.5)
    plt.plot(pvb[order], label="predicted VB", linewidth=1.0, alpha=0.9)
    plt.xlabel("test window (sorted by true wear)")
    plt.ylabel("flank wear (mm)")
    plt.title("CNN-BiLSTM on held-out cutter c6 — wear trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "trajectory.png", dpi=110)
    plt.close()

    plt.figure(figsize=(5, 5))
    lims = [float(min(true_vb.min(), pvb.min())) - 0.01,
            float(max(true_vb.max(), pvb.max())) + 0.01]
    plt.scatter(true_vb, pvb, s=4, alpha=0.4)
    plt.plot(lims, lims, "r--", label="ideal")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("true VB (mm)")
    plt.ylabel("predicted VB (mm)")
    plt.title("Predicted vs true wear (c6)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "scatter.png", dpi=110)
    plt.close()

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(resid, bins=40)
    axes[0].set_xlabel("error (true - pred, mm)")
    axes[0].set_title("Residual histogram")
    axes[1].scatter(true_vb, resid, s=4, alpha=0.4)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("true VB (mm)")
    axes[1].set_ylabel("error (mm)")
    axes[1].set_title("Error vs wear level")
    plt.tight_layout()
    plt.savefig(OUT / "residual.png", dpi=110)
    plt.close()
    print(f"plots saved to {OUT}")


if __name__ == "__main__":
    main()
