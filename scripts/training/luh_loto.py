"""Leave-one-tool-out evaluation of the HGBR tier on LUH/KIT milling wear.

Trains a gradient-boosted regressor (same configuration as the
Vicomtech turning tier) on the LUH signal features for each of the
9 tools held out in turn, and writes ``docs/luh_loto.json`` with
pooled and per-tool RMSE plus baselines. Signals-only and
signals+contact-time variants are both scored; the contact-time-only
linear reference shows how much the signals add.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts/training/luh_loto.py
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


def _fit_predict_hgbr(x_tr, y_tr, x_te):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=63,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def _pooled_rmse(errs: list[np.ndarray]) -> float:
    sq = np.concatenate([e ** 2 for e in errs])
    return float(np.sqrt(sq.mean()))


def main() -> None:
    from hybrid_ai.core.luh_loader import FEATURE_NAMES, load_luh_features

    t0 = time.time()
    X, vb_mm, contact_s, tools, runs = load_luh_features()
    print(f"features: X {X.shape}, {len(set(tools.tolist()))} tools, "
          f"VB {vb_mm.min()*1000:.0f}..{vb_mm.max()*1000:.0f} um ({time.time()-t0:.0f}s)")

    log_c = np.log1p(contact_s)
    variants = {
        "signals": X,
        "signals_logC": np.column_stack([X, log_c]),
    }
    tool_ids = sorted(set(tools.tolist()))

    out: dict = {
        "n_rows": int(len(vb_mm)),
        "n_tools": len(tool_ids),
        "vb_um_range": [float(vb_mm.min() * 1000), float(vb_mm.max() * 1000)],
        "variants": {},
        "note": (
            "LUH/KIT milling (Mendeley zpxs87bjt8), full dataset (6,418 runs); "
            "9-tool leave-one-tool-out; the shipped checkpoint "
            "(models/luh_hgbr.joblib, scripts/training/train_luh_hgbr.py) "
            "refits on all 9 tools after this evaluation"
        ),
    }

    for name, feat in variants.items():
        per_tool: dict[str, float] = {}
        per_tool_base: dict[str, float] = {}
        errs: list[np.ndarray] = []
        base_errs: list[np.ndarray] = []
        for tool in tool_ids:
            te = tools == tool
            tr = ~te
            pred = _fit_predict_hgbr(feat[tr], vb_mm[tr], feat[te])
            errs.append(pred - vb_mm[te])
            base_errs.append(np.full(int(te.sum()), vb_mm[tr].mean()) - vb_mm[te])
            per_tool[str(tool)] = float(np.sqrt(np.mean((pred - vb_mm[te]) ** 2)))
        out["variants"][name] = {
            "pooled_rmse_vb_mm": _pooled_rmse(errs),
            "baseline_train_mean_pooled_rmse_vb_mm": _pooled_rmse(base_errs),
            "per_tool_rmse_vb_mm": per_tool,
        }
        print(f"{name:<14} pooled RMSE {out['variants'][name]['pooled_rmse_vb_mm']:.4f} mm "
              f"(baseline {out['variants'][name]['baseline_train_mean_pooled_rmse_vb_mm']:.4f})")

    c_errs: list[np.ndarray] = []
    for tool in tool_ids:
        te = tools == tool
        tr = ~te
        a, b = np.polyfit(log_c[tr], vb_mm[tr], 1)
        c_errs.append((a * log_c[te] + b) - vb_mm[te])
    out["c_only_linear_reference"] = {
        "pooled_rmse_vb_mm": _pooled_rmse(c_errs),
    }
    c_only = out["c_only_linear_reference"]["pooled_rmse_vb_mm"]
    print(f"{'logC-only lin':<14} pooled RMSE {c_only:.4f} mm")

    out["feature_names"] = list(FEATURE_NAMES)
    dest = ROOT / "docs" / "luh_loto.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
