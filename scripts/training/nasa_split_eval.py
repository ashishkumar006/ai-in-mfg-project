"""Fixed-split evaluation of the production MLP recipe on NASA milling data.

There is no canonical NASA test split, so this holds out every 4th run
(deterministic, spread across the campaign) and trains the production
recipe (1200 synthetic rows + the remaining runs, 120 epochs) fresh, so
the held-out rows were never seen. The recipe is repeated with three
training seeds to measure how stable the result is, and the shipped
model (models/nasa_mlp.pt) is scored on the same rows for reference --
marked as seen-in-training, since it was fitted on all runs.

Writes docs/nasa_split_eval.json. Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts/training/nasa_split_eval.py
"""

from __future__ import annotations

import json
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
from hybrid_ai.tools import _models

BANDS = ((0.0, 0.15, "fresh"), (0.15, 0.30, "steady"), (0.30, 99.0, "worn"))


def _band(v: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= v < hi:
            return name
    return "worn"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    da, db = a - a.mean(), b - b.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / den) if den > 1e-12 else float("nan")


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred[:, 0] - true[:, 0]
    return {
        "rmse_vb_mm": round(float(np.sqrt((err ** 2).mean())), 4),
        "mae_vb_mm": round(float(np.abs(err).mean()), 4),
        "bias_vb_mm": round(float(err.mean()), 4),
        "pearson_r_vb": round(_pearson(pred[:, 0], true[:, 0]), 3),
        "rmse_ra_um": round(float(np.sqrt(((pred[:, 1] - true[:, 1]) ** 2).mean())), 4),
        "within_0p05_mm": round(float((np.abs(err) <= 0.05).mean()), 3),
        "within_0p10_mm": round(float((np.abs(err) <= 0.10).mean()), 3),
    }


def main() -> None:
    Xn, yn, runs, _cases = load_nasa_labeled_features()
    test_mask = runs % 4 == 0
    test_runs = sorted(set(runs[test_mask].tolist()))
    Xte, yte = Xn[test_mask], yn[test_mask]
    Xtr_n, ytr_n = Xn[~test_mask], yn[~test_mask]
    print(f"NASA: {Xn.shape[0]} rows, {len(set(runs.tolist()))} runs")
    print(f"test  = runs {test_runs} ({Xte.shape[0]} rows)")
    print(f"train = {Xtr_n.shape[0]} NASA rows + 1200 synthetic\n")

    t0 = time.time()
    seed_metrics = {}
    seed_preds = {}
    for seed in (42, 43, 44):
        Xs, ys = generate_synthetic_dataset(n_samples=1200, seed=seed)
        model = WearRoughnessModel()
        model.fit(np.concatenate([Xs, Xtr_n]), np.concatenate([ys, ytr_n]),
                  TrainConfig(epochs=120, hidden=(64, 64), seed=seed))
        pred = model.predict(Xte)
        seed_metrics[seed] = _metrics(pred, yte)
        seed_preds[seed] = pred[:, 0]
        print(f"seed {seed}: RMSE VB {seed_metrics[seed]['rmse_vb_mm']:.4f} mm, "
              f"Ra {seed_metrics[seed]['rmse_ra_um']:.4f} um, "
              f"r={seed_metrics[seed]['pearson_r_vb']:+.2f}")

    rm = [m["rmse_vb_mm"] for m in seed_metrics.values()]
    print(f"\nseed-to-seed RMSE spread: {min(rm):.4f} .. {max(rm):.4f} mm")
    pairs = [(42, 43), (42, 44), (43, 44)]
    dmax = max(float(np.abs(seed_preds[a] - seed_preds[b]).mean()) for a, b in pairs)
    print(f"mean pairwise |prediction difference| across seeds: {dmax:.4f} mm")

    main_m = seed_metrics[42]
    print(f"\nmain model (seed 42) vs held-out truth:")
    print(f"  RMSE {main_m['rmse_vb_mm']:.4f} mm | MAE {main_m['mae_vb_mm']:.4f} | "
          f"bias {main_m['bias_vb_mm']:+.4f}")
    print(f"  within +/-0.05 mm: {main_m['within_0p05_mm']:.0%} | "
          f"within +/-0.10 mm: {main_m['within_0p10_mm']:.0%} | "
          f"Pearson r = {main_m['pearson_r_vb']:+.3f}")

    pred = seed_preds[42]
    true = yte[:, 0]
    order = ["fresh", "steady", "worn"]
    agree = 0
    matrix = {a: {b: 0 for b in order} for a in order}
    for t, p in zip(true, pred):
        tb, pb = _band(float(t)), _band(float(p))
        matrix[pb][tb] += 1
        agree += tb == pb
    print(f"\nstate-band agreement (fresh<0.15, steady 0.15-0.3, worn>0.3): "
          f"{agree}/{len(true)} = {agree / len(true):.0%}")
    print("  predicted\\true " + "  ".join(f"{b:>6s}" for b in order))
    for pb in order:
        print(f"  {pb:>14s} " + "  ".join(f"{matrix[pb][tb]:>6d}" for tb in order))

    per_run = []
    print("\nper-run consistency (held-out runs):")
    for r in test_runs:
        m = runs[test_mask] == r
        err = pred[m] - yte[m, 0]
        per_run.append({"run": r, "n": int(m.sum()),
                        "rmse_vb_mm": round(float(np.sqrt((err ** 2).mean())), 4),
                        "bias_vb_mm": round(float(err.mean()), 4)})
        print(f"  run {r:>2d}  n={int(m.sum()):>2d}  RMSE {per_run[-1]['rmse_vb_mm']:.4f}  "
              f"bias {per_run[-1]['bias_vb_mm']:+.4f}")

    shipped = _models.get_mlp()
    shipped_m = _metrics(shipped.predict(Xte), yte)
    print(f"\nshipped model on the same rows (SEEN in training, flattering): "
          f"RMSE {shipped_m['rmse_vb_mm']:.4f} mm, r={shipped_m['pearson_r_vb']:+.2f}")

    out = ROOT / "docs" / "nasa_split_eval.json"
    out.write_text(json.dumps({
        "scheme": "fixed split, runs % 4 == 0 held out",
        "test_runs": test_runs,
        "n_test_rows": int(test_mask.sum()),
        "n_train_nasa_rows": int((~test_mask).sum()),
        "recipe": "1200 synthetic + train runs, 120 epochs, hidden (64, 64)",
        "per_seed": {str(k): v for k, v in seed_metrics.items()},
        "rmse_spread_across_seeds_mm": [min(rm), max(rm)],
        "mean_pairwise_prediction_difference_mm": round(dmax, 4),
        "band_agreement_pct": round(100 * agree / len(true), 1),
        "band_matrix": matrix,
        "per_run": per_run,
        "shipped_model_seen_in_training": shipped_m,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
