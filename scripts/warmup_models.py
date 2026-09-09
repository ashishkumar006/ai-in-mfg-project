"""Pre-flight check: load every dashboard model exactly as app.py does
and run one prediction per model. Fails loudly if anything is broken.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/warmup_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hybrid_ai.ann.model import WearRoughnessDL, WearRoughnessModel, predict_for_point
from hybrid_ai.core.phm_loader import load_phm_windows
from hybrid_ai.core.schemas import OperatingPoint
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_sensor,
)

MODELS_DIR = ROOT / "models"


def main() -> None:
    t0 = time.time()
    mlp = WearRoughnessModel()
    mlp.load(MODELS_DIR / "nasa_mlp.pt")
    print(f"MLP ok ({time.time() - t0:.1f}s)")

    t0 = time.time()
    cnn1d = WearRoughnessDL.load(MODELS_DIR / "nasa_cnn1d.pt")
    cnn_lstm = WearRoughnessDL.load(MODELS_DIR / "nasa_cnn_lstm.pt")
    print(f"CNN1D + CNN-LSTM ok ({time.time() - t0:.1f}s)")

    t0 = time.time()
    big = WearRoughnessDL.load(MODELS_DIR / "phm_cnn_bilstm.pt")
    n_params = sum(p.numel() for p in big.model.parameters())
    print(f"CNN-BiLSTM ok ({time.time() - t0:.1f}s, {n_params / 1e6:.2f}M params)")

    tool = build_default_tool_catalog()[0]
    op = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)
    sensor = generate_synthetic_sensor(250.0, 0.30, 1.75, seed=7)
    pred = predict_for_point(mlp, tool, op, hardness_hb=220.0, sensor=sensor)
    print(f"MLP predicts: V_B={pred.v_b_mm:.3f} mm R_a={pred.ra_um:.3f} um")

    rng = np.random.default_rng(0)
    base = rng.normal(0, sensor.vibration_rms_g, size=9000)
    raw = np.stack([base * 0.9, base, base * 0.8,
                    base * (sensor.current_a / 5.0),
                    base * (sensor.acoustic_rms_db / 60.0),
                    base * (sensor.cutting_force_n / 500.0)])[np.newaxis]
    for name, m in (("CNN1D", cnn1d), ("CNN-LSTM", cnn_lstm)):
        out = m.predict(raw)[0]
        assert np.all(np.isfinite(out)), name
        # same physical clips the dashboard applies
        print(f"{name} predicts: V_B={np.clip(out[0], 0.02, 1.5):.3f} mm "
              f"R_a={np.clip(out[1], 0.2, 6.0):.3f} um")

    x, y, _ = load_phm_windows(split="test")
    j = int(np.argsort(y[:, 0])[len(y) // 2])
    out = big.predict(x[j][np.newaxis])[0]
    print(f"CNN-BiLSTM predicts: V_B={out[0]:.3f} mm (true {y[j, 0]:.3f}) "
          f"R_a={out[1]:.3f} um")
    print("ALL MODELS RUNNING")


if __name__ == "__main__":
    main()
