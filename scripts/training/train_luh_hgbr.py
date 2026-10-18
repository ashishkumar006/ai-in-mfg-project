"""Train the milling-native regressor on the LUH/KIT signal features.

HistGradientBoosting on the 16 signal statistics plus ``log1p`` of the
cumulative tool-contact time, fitted on all nine tools and saved to
``models/luh_hgbr.joblib``. The honest generalisation number is *not*
the in-sample fit: it is the 9-tool leave-one-tool-out RMSE written by
``scripts/training/luh_loto.py`` (``docs/luh_loto.json``), reported
alongside for the record.

Usage (from the project root)::

    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts/training/train_luh_hgbr.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.core.luh_loader import load_luh_features
from hybrid_ai.stages.ann.luh_model import LUHConfig, LUHMillingModel


def main() -> None:
    t0 = time.time()
    x16, vb_mm, contact_s, tools, _runs = load_luh_features()
    x = LUHMillingModel.assemble(x16, contact_s)
    n_tools = len(set(tools.tolist()))
    print(f"features {x.shape} | {n_tools} tools | "
          f"VB {vb_mm.min() * 1000:.0f}..{vb_mm.max() * 1000:.0f} um "
          f"({time.time() - t0:.0f}s)")

    luh_json = ROOT / "docs" / "luh_loto.json"
    if luh_json.exists():
        v = json.loads(luh_json.read_text(encoding="utf-8"))["variants"]["signals_logC"]
        print(f"honest metric (9-tool LOTO): pooled RMSE "
              f"{v['pooled_rmse_vb_mm']:.4f} mm vs baseline "
              f"{v['baseline_train_mean_pooled_rmse_vb_mm']:.4f} mm")
    else:
        print("docs/luh_loto.json missing; run scripts/training/luh_loto.py "
              "for the honest LOTO metric")

    model = LUHMillingModel(config=LUHConfig())
    model.fit(x, vb_mm)
    out = ROOT / "models" / "luh_hgbr.joblib"
    model.save(out)
    print(f"saved {out} in {time.time() - t0:.0f}s")

    in_sample = model.evaluate(x, vb_mm)
    print(f"in-sample RMSE {in_sample['rmse']:.4f} mm "
          "(optimistic; the LOTO number above is the honest one)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
