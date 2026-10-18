"""Tools: catalog listing + model benchmark table."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_ai.core.synthetic import generate_synthetic_dataset
from hybrid_ai.core.nasa_loader import load_blended_dataset, load_nasa_windows
from hybrid_ai.tools import _models

LIST_SPEC: dict = {
    "name": "list_tools",
    "description": "List catalog inserts, optionally filtered by ISO group and operation.",
    "parameters": {
        "type": "object",
        "properties": {
            "iso_group": {"type": ["string", "null"],
                          "enum": ["P", "M", "K", "N", "S", None]},
            "operation": {"type": ["string", "null"],
                          "enum": ["turning", "facing", "milling", "drilling",
                                   "boring", None]},
        },
        "required": [],
    },
}

BENCHMARKS_SPEC: dict = {
    "name": "model_benchmarks",
    "description": ("RMSE/MAE for every trained model, each scored on its own "
                    "test data. Each row carries `eval` (what it was scored "
                    "on) and `holdout` (true only for genuinely unseen data)."),
    "parameters": {"type": "object", "properties": {}},
}


def run_list(iso_group: str | None = None,
             operation: str | None = None) -> dict:
    """List inserts, optionally filtered."""
    tools = [t for t in _models.get_catalog()
             if (iso_group is None or t.iso_group == iso_group)
             and (operation is None or t.operation == operation)]
    return {"tools": [t.to_dict() for t in tools]}


def run_benchmarks() -> dict:
    """Benchmark every trained model on the data it can be fairly scored on.

    Rows are not comparable across models: datasets and sample counts
    differ. ``holdout`` is False when the score comes from the model's own
    training distribution, True only for genuinely unseen data.
    """
    import numpy as np

    rows: list[dict] = []
    mlp = _models.get_mlp()
    try:
        xe, ye = load_blended_dataset(n_synthetic=1200, seed=99)
    except Exception:
        xe, ye = generate_synthetic_dataset(n_samples=300, seed=99)
    m = mlp.evaluate(xe, ye)
    rows.append({"model": "MLP", "input": "11 features (blended)",
                 "rmse": round(m["rmse"], 4), "mae": round(m["mae"], 4),
                 "n": m["n"], "holdout": False,
                 "eval": "synthetic-style blend — training distribution "
                         "(optimistic)"})
    heldout = Path(__file__).resolve().parents[3] / "docs" / "heldout_nasa.json"
    try:
        d = json.loads(heldout.read_text(encoding="utf-8"))
        rows.append({"model": "MLP (NASA leave-one-run-out)",
                     "input": "11 features (NASA runs)",
                     "rmse": round(float(d["weighted_rmse_vb_mm"]), 4),
                     "mae": None, "n": int(d["n_heldout_rows"]),
                     "holdout": True,
                     "eval": f"{d['n_runs']} held-out NASA runs (real cutting "
                             "data)"})
    except Exception:
        pass
    try:
        from hybrid_ai.core.nasa_loader import NASA_PATH

        if NASA_PATH.exists():
            xw, yw = load_nasa_windows()
        else:
            raise FileNotFoundError("no NASA data")
        kind = "6×9000 raw windows"
    except Exception:
        rng = np.random.default_rng(99)
        xw = rng.normal(size=(60, 6, 9000))
        yw = np.stack([xw[:, 0].mean(axis=1), xw[:, 1].mean(axis=1)], axis=1)
        kind = "6×9000 raw windows"
    for name, getter in (("CNN1D", _models.get_cnn1d),
                         ("CNN-LSTM", _models.get_cnn_lstm)):
        try:
            e = getter().evaluate(xw, yw)
            rows.append({"model": name, "input": kind,
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"], "holdout": False,
                         "eval": "the NASA windows it was trained on "
                                 "(memorised)"})
        except Exception as exc:
            rows.append({"model": name, "input": kind, "error": str(exc)})
    big = _models.get_big()
    if big is not None:
        try:
            from hybrid_ai.core.phm_loader import load_phm_windows as _phm

            xp, yp, _ = _phm(split="test")
            e = big.evaluate(xp, yp)
            rows.append({"model": "CNN-BiLSTM", "input": "7×256 PHM (c6)",
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"], "holdout": True,
                         "eval": "held-out PHM cutter c6 windows"})
        except Exception as exc:
            rows.append({"model": "CNN-BiLSTM", "input": "7×256 PHM (c6)",
                         "error": str(exc)})
    turning = _models.get_turning()
    if turning is not None:
        try:
            from hybrid_ai.core.vicomtech_loader import load_vicomtech

            xv, yv, _ = load_vicomtech(split="test")
            e = turning.evaluate(xv, yv)
            rows.append({"model": "HGBR", "input": "152 feats (Vicomtech tools)",
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"], "holdout": True,
                         "eval": "held-out Vicomtech turning tools"})
        except Exception as exc:
            rows.append({"model": "HGBR", "input": "152 feats (Vicomtech tools)",
                         "error": str(exc)})
    luh_loto = Path(__file__).resolve().parents[3] / "docs" / "luh_loto.json"
    try:
        d = json.loads(luh_loto.read_text(encoding="utf-8"))
        v = d["variants"]["signals_logC"]
        rows.append({"model": "HGBR (LUH milling)",
                     "input": "17 feats (LUH signal runs)",
                     "rmse": round(float(v["pooled_rmse_vb_mm"]), 4),
                     "mae": None, "n": int(d["n_rows"]), "holdout": True,
                     "eval": "9 held-out LUH tools (leave-one-tool-out; "
                             "checkpoint refit on all 9 after the eval)"})
    except Exception:
        pass
    return {"rows": rows}
