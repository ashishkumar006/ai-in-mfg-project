"""Real-data scoreboard: every trained model on its held-out real dataset.

Loads the shipped checkpoints and evaluates each on the genuinely unseen
split of the real dataset it targets (NASA runs, PHM cutter c6,
Vicomtech tools 15+16, PHM+NUAA c6+W9, Haas tools 9+105, AISI 5140 L9).
Reports RMSE against the train-mean baseline and the percent of variance
explained (skill = 1 - (rmse/baseline)^2). No training happens here.

Two-target models (VB in mm + Ra in um) are scored the way
WearRoughnessDL.evaluate scores them: one RMSE over both outputs, so
their rows are mixed-unit and only comparable to their own baseline.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe scripts/checks/real_data_scoreboard.py
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


def mixed_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mean_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    return mixed_rmse(y_test, np.asarray(y_train).mean(axis=0))


def skill(rmse: float, baseline: float) -> str:
    if not np.isfinite(baseline) or baseline <= 0:
        return "-"
    return f"{max(0.0, 1.0 - (rmse / baseline) ** 2) * 100:.0f}%"


def main() -> None:
    t0 = time.time()
    rows: list[dict] = []

    from hybrid_ai.tools import _models

    mlp = _models.get_mlp()
    from hybrid_ai.core.nasa_loader import load_blended_dataset, load_nasa_windows

    xe, ye = load_blended_dataset(n_synthetic=1200, seed=99)
    m = mlp.evaluate(xe, ye)
    rows.append({"dataset": "synthetic+NASA blend", "model": "MLP (11 feat)",
                 "holdout": "no", "n": m["n"], "rmse": m["rmse"],
                 "baseline": float("nan"),
                 "note": "training distribution, VB+Ra mixed (optimistic)"})

    d = json.loads((ROOT / "docs" / "heldout_nasa.json").read_text(encoding="utf-8"))
    rows.append({"dataset": "NASA runs (LORO)", "model": "MLP (11 feat)",
                 "holdout": "yes", "n": d["n_heldout_rows"],
                 "rmse": float(d["weighted_rmse_vb_mm"]),
                 "baseline": float(d["baseline_train_mean_rmse_vb_mm"]),
                 "note": f"{d['n_runs']} runs, VB only; crippled features (Vc const)"})

    xw, yw = load_nasa_windows()
    for name, getter in (("CNN1D", _models.get_cnn1d),
                         ("CNN-LSTM", _models.get_cnn_lstm)):
        e = getter().evaluate(xw, yw)
        rows.append({"dataset": "NASA windows", "model": name,
                     "holdout": "no (memorised)", "n": e["n"], "rmse": e["rmse"],
                     "baseline": float("nan"),
                     "note": "scored on its own training set; legacy tier"})

    from hybrid_ai.core.phm_loader import load_phm_windows

    xtr, ytr, _ = load_phm_windows(split="train")
    xte, yte, _ = load_phm_windows(split="test")
    big = _models.get_big()
    e = big.evaluate(xte, yte)
    rows.append({"dataset": "PHM 2010 cutter c6", "model": "CNN-BiLSTM (7ch)",
                 "holdout": "yes", "n": e["n"], "rmse": e["rmse"],
                 "baseline": mean_baseline(ytr, yte),
                 "note": "unseen cutter, VB+Ra mixed"})

    from hybrid_ai.core.vicomtech_loader import load_vicomtech

    vtr = load_vicomtech(split="train")
    vte = load_vicomtech(split="test")
    turn = _models.get_turning()
    e = turn.evaluate(vte[0], vte[1])
    rows.append({"dataset": "Vicomtech tools 15+16", "model": "HGBR (152 feat)",
                 "holdout": "yes", "n": e["n"], "rmse": e["rmse"],
                 "baseline": mean_baseline(vtr[1], vte[1]),
                 "note": "unseen tools, VB+Ra mixed"})

    from hybrid_ai.core.unified_loader import load_unified_windows
    from hybrid_ai.stages.ann.model import WearRoughnessDL

    utr = load_unified_windows(split="train")
    ute = load_unified_windows(split="test")
    uni = WearRoughnessDL.load(ROOT / "models" / "unified_cnn_bilstm.pt")
    e = uni.evaluate(ute[0], ute[1])
    rows.append({"dataset": "PHM+NUAA pooled", "model": "CNN-BiLSTM (3ch)",
                 "holdout": "yes", "n": e["n"], "rmse": e["rmse"],
                 "baseline": mean_baseline(utr[1], ute[1]),
                 "note": "c6 + W9; pooled training, W9 stays hard"})

    from hybrid_ai.core.haas_loader import load_haas_sequences
    from hybrid_ai.stages.ann.rul_model import RulSequenceModel

    htr = load_haas_sequences(split="train")
    hte = load_haas_sequences(split="test")
    gru = RulSequenceModel.load(ROOT / "models" / "haas_rul_gru.pt")
    ytr_all = np.concatenate([y for _, y, _ in htr])
    errs, base_errs, ns = [], [], []
    for x, y, _tool in hte:
        errs.append(mixed_rmse(y, gru.predict_seq(x)))
        base_errs.append(mixed_rmse(y, ytr_all.mean()))
        ns.append(len(y))
    w = np.asarray(ns, dtype=float)
    rmse = float(np.average(errs, weights=w))
    base = float(np.average(base_errs, weights=w))
    rows.append({"dataset": "Haas tools 9+105", "model": "GRU (RUL)",
                 "holdout": "yes", "n": int(w.sum()), "rmse": rmse,
                 "baseline": base,
                 "note": "cycles remaining, not VB; 2 longest lives held out"})

    from hybrid_ai.core.aisi5140_loader import load_aisi5140
    from hybrid_ai.core.synthetic import ra_model_um

    luh_json = ROOT / "docs" / "luh_loto.json"
    if not luh_json.exists():
        raise FileNotFoundError(
            "docs/luh_loto.json missing; run scripts/training/luh_loto.py "
            "(needs LUH data from scripts/checks/download_luh_subset.py)."
        )
    luh = json.loads(luh_json.read_text(encoding="utf-8"))
    v = luh["variants"]["signals_logC"]
    rows.append({
        "dataset": "LUH milling (9-tool LOTO)", "model": "HGBR (16 feat+logC)",
        "holdout": "yes", "n": luh["n_rows"],
        "rmse": v["pooled_rmse_vb_mm"],
        "baseline": v["baseline_train_mean_pooled_rmse_vb_mm"],
        "note": "cast-iron milling, VB 3-160 um, full dataset (6,418 runs); "
                f"signals-only {luh['variants']['signals']['pooled_rmse_vb_mm']:.4f} "
                f"beats logC-only {luh['c_only_linear_reference']['pooled_rmse_vb_mm']:.4f}; "
                "labels interpolated every ~40 runs",
    })

    xa, ya = load_aisi5140()
    pred_a = mlp.predict(xa)
    pred_a[:, 0] = np.clip(pred_a[:, 0], 0.02, 1.5)
    pred_a[:, 1] = np.clip(pred_a[:, 1], 0.2, 6.0)
    row = {"dataset": "AISI 5140 turning (L9)", "model": "MLP (11 feat)",
           "holdout": "n/a (zero-shot)", "n": len(ya),
           "rmse": mixed_rmse(ya, pred_a),
           "baseline": mixed_rmse(ya, np.asarray(ye).mean(axis=0)),
           "note": "external L9 set, measured VB+Ra; see AISI detail below"}

    aisi_detail = {
        "vb_rmse": mixed_rmse(ya[:, 0], pred_a[:, 0]),
        "vb_spread": float(np.std(ya[:, 0])),
        "ra_rmse": mixed_rmse(ya[:, 1], pred_a[:, 1]),
        "ra_spread": float(np.std(ya[:, 1])),
        "test_mean_mixed": mixed_rmse(ya, ya.mean(axis=0)),
        "oracle_ra_rmse": mixed_rmse(
            ya[:, 1],
            np.array([ra_model_um(f, 0.8, vb, "P") for f, vb in zip(xa[:, 1], ya[:, 0])]),
        ),
    }
    row["note"] = (
        f"external L9 set, measured VB+Ra; VB {aisi_detail['vb_rmse']:.3f} vs "
        f"{aisi_detail['vb_spread']:.3f} spread, all 9 under-predicted "
        "(no wear-state input); Ra detail in AISI block"
    )
    rows.append(row)

    head = (f"{'dataset':<26} {'model':<17} {'hold':<14} {'n':>5} "
            f"{'RMSE':>8} {'base':>8} {'skill':>6}")
    print(head)
    print("-" * len(head))
    for r in rows:
        base_s = "-" if not np.isfinite(r["baseline"]) else f"{r['baseline']:.4f}"
        print(f"{r['dataset']:<26} {r['model']:<17} {r['holdout']:<14} "
              f"{r['n']:>5} {r['rmse']:>8.4f} {base_s:>8} "
              f"{skill(r['rmse'], r['baseline']):>6}")
    print()
    for r in rows:
        print(f"- {r['dataset']} / {r['model']}: {r['note']}")
    print()
    print("AISI 5140 detail (9 rows, measured labels; clipping as deployed):")
    print(f"  MLP zero-shot:   VB RMSE {aisi_detail['vb_rmse']:.3f} mm "
          f"(spread {aisi_detail['vb_spread']:.3f})  |  "
          f"Ra RMSE {aisi_detail['ra_rmse']:.3f} um (spread {aisi_detail['ra_spread']:.3f})")
    print(f"  constants:       train-mean mixed {rows[-1]['baseline']:.3f}  |  "
          f"test-mean mixed {aisi_detail['test_mean_mixed']:.3f}")
    print(f"  Ra model (oracle VB): {aisi_detail['oracle_ra_rmse']:.3f} um RMSE")
    print("  -> params-only MLP cannot see accumulated wear; the roughness model")
    print("     misses the feed-independent Ra floor seen in real turning.")
    print(f"\n{len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
