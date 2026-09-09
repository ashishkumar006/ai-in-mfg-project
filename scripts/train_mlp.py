"""Train the blended MLP (reproduces models/nasa_mlp.pt).

11-feature schema: synthetic (wide envelope) + NASA-mapped rows.
CPU-only, seconds.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/train_mlp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.stages.ann.model import TrainConfig, WearRoughnessModel
from hybrid_ai.core.nasa_loader import load_blended_dataset


def main() -> None:
    t0 = time.time()
    X, y = load_blended_dataset(n_synthetic=1200, seed=42)
    print(f"blended: {X.shape}")
    model = WearRoughnessModel()
    history = model.fit(X, y, TrainConfig(epochs=120, hidden=(64, 64)))
    print(f"trained in {time.time() - t0:.1f}s, final loss {history['final_loss']:.4f}")
    print("train eval:", model.evaluate(X, y))
    out = ROOT / "models" / "nasa_mlp.pt"
    model.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
