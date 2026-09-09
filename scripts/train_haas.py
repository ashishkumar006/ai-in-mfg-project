"""Train Haas RUL models (reproduces models/haas_rul_gru.pt).

A GRU over each tool's cycle history predicts remaining cycles.
Split by tool: train 10 tools / val 2 / test tools 9+105. A
single-cycle HGBR is reported as the baseline it must beat.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/train_haas.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from hybrid_ai.stages.ann.rul_model import RulModel, RulSeqConfig, RulSequenceModel
from hybrid_ai.core.haas_loader import load_haas, load_haas_sequences


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.ravel(a) - np.ravel(b)) ** 2)))


def flat(seqs):
    return np.concatenate([y for _, y, _ in seqs]), None


def main() -> None:
    torch.set_num_threads(12)
    t0 = time.time()
    tr = load_haas_sequences(split="train")
    va = load_haas_sequences(split="val")
    te = load_haas_sequences(split="test")
    ntr = sum(len(y) for _, y, _ in tr)
    print(f"train {len(tr)} tools/{ntr} cycles | val {len(va)} tools | "
          f"test {len(te)} tools {[t for _, _, t in te]}")

    ytr_all = np.concatenate([y for _, y, _ in tr])
    yte_all = np.concatenate([y for _, y, _ in te])
    print(f"mean-baseline test RMSE: {rmse(yte_all, ytr_all.mean()):.2f} cycles")

    model = RulSequenceModel(config=RulSeqConfig())
    history = model.fit([(x, y) for x, y, _ in tr],
                        [(x, y) for x, y, _ in va])
    out = ROOT / "models" / "haas_rul_gru.pt"
    model.save(out)
    print(f"saved {out} in {time.time() - t0:.0f}s "
          f"({history['epochs_run']} epochs)")

    errs, errs0, ns = [], [], []
    for x, y, tool in te:
        p = model.predict_seq(x)
        errs.append(rmse(y, p))
        errs0.append(rmse(y, ytr_all.mean()))
        ns.append(len(y))
        print(f"  tool {tool}: {len(y):3d} cycles rmse={errs[-1]:6.2f} "
              f"(baseline {errs0[-1]:6.2f})")
    w = np.array(ns, dtype=float)
    print(f"test weighted RMSE {float(np.average(errs, weights=w)):.2f} "
          f"(baseline {float(np.average(errs0, weights=w)):.2f})")

    Xb, yb, _ = load_haas(split="train")
    Xe, ye, _ = load_haas(split="test")
    snap = RulModel()
    snap.fit(Xb, yb)
    print(f"single-cycle HGBR baseline: test RMSE "
          f"{snap.evaluate(Xe, ye)['rmse']:.2f} cycles")


if __name__ == "__main__":
    main()
