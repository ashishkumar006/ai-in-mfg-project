"""Honest held-out evaluation of the blended MLP on NASA milling runs.

The blended MLP (models/nasa_mlp.pt) is trained on synthetic rows *plus*
NASA rows. Its headline RMSE on a random blended holdout is flattered by
the synthetic majority, and by NASA rows that were in the training set.

This recipe answers the harder question the deck implies: how well does
the model predict a NASA run it has **never seen**? It does leave-one-run-
out (LORO) cross-validation over the 23 NASA runs, training a fresh MLP on
the synthetic rows plus every *other* run, then scoring the held-out run.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/training/heldout_nasa.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.core.nasa_loader import load_nasa_labeled_features
from hybrid_ai.core.synthetic import generate_synthetic_dataset
from hybrid_ai.stages.ann.model import TrainConfig, WearRoughnessModel


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    Xn, yn, runs, _ = load_nasa_labeled_features()
    Xs, ys = generate_synthetic_dataset(n_samples=1200, seed=42)
    print(f"NASA rows: {Xn.shape[0]} across runs {sorted(set(runs.tolist()))}")
    print(f"synthetic rows: {Xs.shape[0]}")

    cfg = TrainConfig(epochs=80, hidden=(64, 64))
    rows: list[tuple[int, int, float, float]] = []
    t0 = time.time()
    for r in sorted(set(runs.tolist())):
        te = runs == r
        tr_nasa = ~te
        Xtr = np.concatenate([Xs, Xn[tr_nasa]], axis=0)
        ytr = np.concatenate([ys, yn[tr_nasa]], axis=0)
        model = WearRoughnessModel()
        model.fit(Xtr, ytr, cfg)
        pred = model.predict(Xn[te])
        rmse_vb = _rmse(pred[:, 0], yn[te][:, 0])
        rmse_ra = _rmse(pred[:, 1], yn[te][:, 1])
        rows.append((r, int(te.sum()), rmse_vb, rmse_ra))

    # A trivial baseline puts the same honest test in context.
    y_mean = yn[:, 0].mean()
    base_vb = _rmse(np.full(Xn.shape[0], y_mean), yn[:, 0])

    total = sum(n for _, n, _, _ in rows)
    w_vb = sum(n * v for _, n, v, _ in rows) / total
    worst = max(rows, key=lambda t: t[2])
    print(f"\nLORO over {len(rows)} runs in {time.time() - t0:.1f}s "
          f"({total} held-out rows)")
    print(f"{'run':>4} {'n':>3} {'RMSE VB':>9} {'RMSE Ra':>9}")
    for r, n, v, a in rows:
        print(f"{r:>4} {n:>3} {v:>9.4f} {a:>9.4f}")
    print(f"\nweighted RMSE VB (held-out): {w_vb:.4f} mm")
    print(f"worst run {worst[0]} (n={worst[1]}): {worst[2]:.4f} mm")
    print(f"train-mean baseline RMSE VB: {base_vb:.4f} mm")

    out = ROOT / "docs" / "heldout_nasa.json"
    import json

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scheme": "leave-one-run-out", "n_runs": len(rows),
        "n_heldout_rows": total, "weighted_rmse_vb_mm": round(w_vb, 4),
        "worst_run": {"run": worst[0], "n": worst[1],
                      "rmse_vb_mm": round(worst[2], 4)},
        "baseline_train_mean_rmse_vb_mm": round(base_vb, 4),
        "per_run": [
            {"run": r, "n": n, "rmse_vb_mm": round(v, 4),
             "rmse_ra_um": round(a, 4)}
            for r, n, v, a in rows
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
