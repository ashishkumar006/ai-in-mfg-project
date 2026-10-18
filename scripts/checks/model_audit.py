"""Audit the production MLP: what it trained on, and whether it over- or underfits.

Reproduces `models/nasa_mlp.pt`'s recipe (1200 synthetic + NASA-mapped rows),
then measures per-target error (VB in mm, Ra in um separately) on:

- fresh synthetic rows (same distribution, new seed)
- NASA rows the shipped model saw during training
- NASA runs held out by run id (never seen)
- a synthetic-only model and a NASA-only model, to expose the transfer gap

Writes `docs/model_audit.json`. CPU-only, under a minute.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts/checks/model_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_ai.core.nasa_loader import (
    NASA_PATH,
    load_nasa_labeled_features,
)
from hybrid_ai.core.synthetic import generate_synthetic_dataset
from hybrid_ai.stages.ann.model import TrainConfig, WearRoughnessModel
from hybrid_ai.tools import _models

HIDDEN = (64, 64)
EPOCHS = 120


def _vb_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred[:, 0] - y_true[:, 0]
    return {
        "rmse_vb": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae_vb": round(float(np.mean(np.abs(err))), 4),
        "bias_vb": round(float(np.mean(err)), 4),
        "n": int(err.shape[0]),
    }


def _ra_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred[:, 1] - y_true[:, 1]
    return {
        "rmse_ra": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae_ra": round(float(np.mean(np.abs(err))), 4),
        "n": int(err.shape[0]),
    }


def main() -> None:
    t0 = time.time()
    out: dict = {}

    # ---- dataset composition -------------------------------------------
    import pandas as pd

    df = pd.read_parquet(NASA_PATH).dropna(subset=["VB"]).reset_index(drop=True)
    x_syn, y_syn = generate_synthetic_dataset(1200, seed=42)
    x_nasa, y_nasa, runs, cases = load_nasa_labeled_features()
    out["datasets"] = {
        "synthetic_rows": int(x_syn.shape[0]),
        "nasa_rows_used": int(x_nasa.shape[0]),
        "nasa_raw_rows": int(df.shape[0]),
        "nasa_runs": int(df["run"].nunique()),
        "nasa_materials": {str(int(k)): int(v) for k, v in df["material"].value_counts().items()},
        "nasa_feed_range": [float(df["feed"].min()), float(df["feed"].max())],
        "nasa_doc_range": [float(df["DOC"].min()), float(df["DOC"].max())],
        "nasa_vb_dist": {
            "min": float(df["VB"].min()),
            "median": float(df["VB"].median()),
            "max": float(df["VB"].max()),
        },
        "synthetic_vb_mean": round(float(y_syn[:, 0].mean()), 4),
        "synthetic_intrinsic_noise_vb": round(0.08 * float(y_syn[:, 0].mean()), 4),
        "real_share_of_blend": round(x_nasa.shape[0] / (x_nasa.shape[0] + 1200), 4),
        "nasa_speed_feature": "constant per material (150 M / 200 K m/min default, not measured)",
        "nasa_ra_label": "imputed from feed+VB, not measured",
    }

    # ---- shipped model ---------------------------------------------------
    mlp = _models.get_mlp()
    x_syn_hold, y_syn_hold = generate_synthetic_dataset(5000, seed=123)
    p_syn = mlp.predict(x_syn_hold)
    out["shipped_on_synthetic_holdout"] = {**_vb_stats(y_syn_hold, p_syn),
                                           **_ra_stats(y_syn_hold, p_syn)}
    p_nasa = mlp.predict(x_nasa)
    out["shipped_on_nasa_train_rows"] = _vb_stats(y_nasa, p_nasa)
    mask = runs % 4 == 0
    ho_true, ho_pred = y_nasa[mask], p_nasa[mask]
    out["shipped_on_nasa_heldout_runs"] = _vb_stats(ho_true, ho_pred)
    mu = float(y_nasa[~mask, 0].mean())
    base_err = ho_true[:, 0] - mu
    out["baseline_mean_on_heldout"] = {
        "rmse_vb": round(float(np.sqrt(np.mean(base_err ** 2))), 4),
        "constant_vb_mm": round(mu, 4),
        "n": int(mask.sum()),
    }
    r = float(np.corrcoef(ho_true[:, 0], ho_pred[:, 0])[0, 1])
    out["shipped_on_nasa_heldout_runs"]["corr_vb"] = round(r, 3)

    # ---- progressive fits: over/underfit curves --------------------------
    tr_x = np.concatenate([x_syn, x_nasa[~mask]])
    tr_y = np.concatenate([y_syn, y_nasa[~mask]])
    curves = []
    for ep in (15, 30, 60, 120):
        m = WearRoughnessModel()
        hist = m.fit(tr_x, tr_y, TrainConfig(epochs=ep, hidden=HIDDEN, seed=42))
        row = {
            "epochs": ep,
            "final_train_mse_normalised": round(hist["final_loss"], 5),
            "train_rows": _vb_stats(tr_y, m.predict(tr_x)),
            "synthetic_holdout": _vb_stats(y_syn_hold, m.predict(x_syn_hold)),
            "nasa_heldout_runs": _vb_stats(ho_true, m.predict(x_nasa[mask])),
        }
        curves.append(row)
        print("epochs", ep, "train", row["train_rows"]["rmse_vb"],
              "syn-hold", row["synthetic_holdout"]["rmse_vb"],
              "nasa-hold", row["nasa_heldout_runs"]["rmse_vb"])
    out["fit_curves"] = curves

    # ---- transfer: synthetic-only vs NASA-only ---------------------------
    m_syn = WearRoughnessModel()
    m_syn.fit(x_syn, y_syn, TrainConfig(epochs=EPOCHS, hidden=HIDDEN, seed=42))
    out["synthetic_only_on_nasa_heldout"] = _vb_stats(ho_true, m_syn.predict(x_nasa[mask]))

    m_nasa = WearRoughnessModel()
    m_nasa.fit(x_nasa[~mask], y_nasa[~mask], TrainConfig(epochs=EPOCHS, hidden=HIDDEN, seed=42))
    out["nasa_only_on_nasa_heldout"] = _vb_stats(ho_true, m_nasa.predict(x_nasa[mask]))

    out["wall_time_s"] = round(time.time() - t0, 1)
    dest = ROOT / "docs" / "model_audit.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
