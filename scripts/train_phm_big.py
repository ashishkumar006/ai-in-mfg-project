"""Train the flagship CNN-BiLSTM on PHM 2010 (reproduces models/phm_cnn_bilstm.pt).

Protocol: train on cutters c1+c4 with early stopping on a
non-overlapping val subset, final report on held-out cutter c6.
CPU-only, a few minutes on a 12-thread laptop.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/train_phm_big.py
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

from hybrid_ai.stages.ann.model import DLConfig, WearRoughnessDL
from hybrid_ai.core.phm_loader import load_phm_windows


def main() -> None:
    torch.set_num_threads(12)
    Xpool, ypool, _ = load_phm_windows(window=256, stride=16, split="train")
    # non-overlapping val subset: stride is 16, window 256, so every 16th
    # window shares no samples with another 16th window. Same cutters as
    # train (fine for early stopping), final test stays unseen cutter c6.
    idx = np.arange(len(Xpool))
    va_mask = (idx % 16) == 0
    Xtr, ytr = Xpool[~va_mask], ypool[~va_mask]
    Xva, yva = Xpool[va_mask], ypool[va_mask]
    Xte, yte, _ = load_phm_windows(window=256, stride=16, split="test")
    print(f"train: {Xtr.shape} | val: {Xva.shape} (non-overlapping) | "
          f"test c6: {Xte.shape}")

    base = float(np.sqrt(np.mean((yte - ytr.mean(axis=0)) ** 2)))
    print(f"mean-baseline test RMSE: {base:.4f}")

    cfg = DLConfig(model_type="cnn_bilstm", epochs=30, batch_size=64,
                   n_channels=7, window_len=256, lstm_hidden=128,
                   lstm_layers=2, learning_rate=1e-3, dropout=0.2,
                   weight_decay=1e-5, patience=6)
    model = WearRoughnessDL(config=cfg)
    n_params = sum(p.numel() for p in model._build(Xtr).parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    t0 = time.time()
    history = model.fit(Xtr, ytr, val_data=(Xva, yva))
    print(f"ran {history['epochs_run']} epochs in {time.time() - t0:.0f}s "
          f"(early stop: {history['stopped_early']}, "
          f"best val: {history['best_val_loss']:.4f})")
    out = ROOT / "models" / "phm_cnn_bilstm.pt"
    model.save(out)
    print(f"saved {out}")
    print("held-out cutter c6:", model.evaluate(Xte, yte))


if __name__ == "__main__":
    main()
