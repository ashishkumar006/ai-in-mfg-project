"""Train a CNN-BiLSTM on the pooled PHM+NUAA 3-channel windows.

Protocol: train on 8 experiments, early-stop on 2 (W7/W8), final
report on held-out c6 + W9 (in-family and out-of-family).
Saves models/unified_cnn_bilstm.pt.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/train_unified.py
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

from hybrid_ai.ann.model import DLConfig, WearRoughnessDL
from hybrid_ai.core.unified_loader import load_unified_windows


def main() -> None:
    torch.set_num_threads(12)
    Xtr, ytr, gtr = load_unified_windows(split="train")
    Xva, yva, gva = load_unified_windows(split="val")
    Xte, yte, gte = load_unified_windows(split="test")
    print(f"train: {Xtr.shape} | val: {Xva.shape} | test: {Xte.shape}",
          flush=True)
    print(f"test groups: {sorted(set(gte.tolist()))}", flush=True)

    base = float(np.sqrt(np.mean((yte - ytr.mean(axis=0)) ** 2)))
    print(f"mean-baseline test RMSE: {base:.4f}", flush=True)

    cfg = DLConfig(model_type="cnn_bilstm", epochs=10, batch_size=128,
                   n_channels=3, window_len=256, lstm_hidden=128,
                   lstm_layers=2, learning_rate=1e-3, dropout=0.2,
                   weight_decay=1e-5, patience=3, verbose=True)
    model = WearRoughnessDL(config=cfg)
    t0 = time.time()
    history = model.fit(Xtr, ytr, val_data=(Xva, yva))
    print(f"ran {history['epochs_run']} epochs in {time.time() - t0:.0f}s "
          f"(early stop: {history['stopped_early']})", flush=True)
    out = ROOT / "models" / "unified_cnn_bilstm.pt"
    model.save(out)
    print(f"saved {out}", flush=True)
    print("test overall:", model.evaluate(Xte, yte), flush=True)
    for grp in sorted(set(gte.tolist())):
        mask = gte == grp
        print(f"  test {grp}:", model.evaluate(Xte[mask], yte[mask]), flush=True)


if __name__ == "__main__":
    main()
