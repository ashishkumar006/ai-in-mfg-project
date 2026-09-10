"""Transfer proof: c1-pretrained CNN-BiLSTM fine-tuned on 400 c4 windows
vs from-scratch on the same 400. Evaluates on held-out c4 remainder
(same distribution) and unseen cutter c6 (distribution shift).

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/training/transfer_proof.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from hybrid_ai.stages.ann.model import DLConfig, WearRoughnessDL, fine_tune_head
from hybrid_ai.core.phm_loader import load_phm_windows


def run_seed(seed: int, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> None:
    c1 = (g == "c1")
    c4 = (g == "c4")
    c6 = (g == "c6")
    rng = np.random.default_rng(seed)
    small = np.zeros(len(g), dtype=bool)
    small[rng.choice(np.where(c4)[0], 400, replace=False)] = True
    rest = c4 & ~small
    base_cfg = dict(model_type="cnn_bilstm", batch_size=64, n_channels=7,
                    window_len=256, lstm_hidden=128, lstm_layers=2,
                    learning_rate=1e-3, dropout=0.2)
    with tempfile.TemporaryDirectory() as tmp:
        src_path = str(Path(tmp) / "src.pt")
        src = WearRoughnessDL(config=DLConfig(epochs=8, seed=seed, **base_cfg))
        src.fit(X[c1], y[c1])
        src.save(src_path)
        ft = fine_tune_head(src_path, X[small], y[small],
                            epochs=5, unfreeze_after=3, seed=seed)
    sc = WearRoughnessDL(config=DLConfig(epochs=5, seed=seed, **base_cfg))
    sc.fit(X[small], y[small])
    print(f"seed {seed}: fine-tuned c4-rest={ft.evaluate(X[rest], y[rest])['rmse']:.4f} "
          f"c6={ft.evaluate(X[c6], y[c6])['rmse']:.4f} | "
          f"scratch c4-rest={sc.evaluate(X[rest], y[rest])['rmse']:.4f} "
          f"c6={sc.evaluate(X[c6], y[c6])['rmse']:.4f}")


def main() -> None:
    torch.set_num_threads(12)
    t0 = time.time()
    X, y, g = load_phm_windows(window=256, stride=16)
    print(f"windows: {len(X)} (c1={(g == 'c1').sum()}, "
          f"c4={(g == 'c4').sum()}, c6={(g == 'c6').sum()})")
    for seed in (42, 7, 123):
        run_seed(seed, X, y, g)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
