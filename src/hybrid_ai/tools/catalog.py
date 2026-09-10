"""Tools: catalog listing + model benchmark table."""

from __future__ import annotations

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
    "description": "Held-out metrics (RMSE/MAE) for every trained model.",
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
    """Benchmark every trained model on its held-out data."""
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
                 "n": m["n"]})
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
                         "n": e["n"]})
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
                         "n": e["n"]})
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
                         "n": e["n"]})
        except Exception as exc:
            rows.append({"model": "HGBR", "input": "152 feats (Vicomtech tools)",
                         "error": str(exc)})
    return {"rows": rows}
