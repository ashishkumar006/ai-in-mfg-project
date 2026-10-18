"""Loader for the AISI 5140 turning dataset (Sensors 2020, PMC7472038).

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC7472038/ (CC BY).
Nine Taguchi L9 turning runs with *measured* flank wear (VB, mm) and
*measured* surface roughness (Ra, um, stylus) alongside cutting
parameters (v_c, f, a_p) and sensor trace summaries (force, AE,
temperature, motor current). The 9 rows are transcribed in
``data/raw/aisi5140_turning.csv``.

Typical usage::

    from hybrid_ai.core.aisi5140_loader import load_aisi5140
    X, y = load_aisi5140()
    # X: (9, 11) in the synthetic/NASA 11-dim schema
    # y: (9, 2) -> [VB_mm, Ra_um], both measured (unlike NASA)

Assumptions (not published in the paper, kept as module constants so
callers can document them):

- Insert WCMT 06T308 P25 -> nose radius 0.8 mm; rake not published,
  assumed 0 deg.
- AISI 5140 (alloy steel, ISO P); hardness not published, assumed
  235 HB.
- v_c levels 150/200/300 m/min (the paper's Table 3 prints 330 for
  level three; the text and figure use 300).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AISI5140_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "aisi5140_turning.csv"

AISI5140_NOSE_RADIUS_MM = 0.8
AISI5140_RAKE_ANGLE_DEG = 0.0
AISI5140_HARDNESS_HB = 235.0
AISI5140_ISO_GROUP = "P"
_ISO_CODES = ("P", "M", "K", "N", "S")


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"AISI 5140 dataset not found at {path}. Transcribe Table 4 from "
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7472038/ into "
            "data/raw/aisi5140_turning.csv."
        )
    return path


def load_aisi5140(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the 9 AISI 5140 runs and return ``(X, y)`` arrays.

    Parameters
    ----------
    path:
        CSV location. Defaults to ``data/raw/aisi5140_turning.csv``.

    Returns
    -------
    X:
        ``(9, 11)`` float64 matrix in the synthetic/NASA schema:
        ``[V_c, f, a_p, nose_radius, rake_angle, iso_P..iso_S,
        hardness]``.
    y:
        ``(9, 2)`` float64 ``[VB_mm, Ra_um]``, both measured.
    """
    import pandas as pd

    src = Path(path) if path is not None else _check_file(AISI5140_PATH)
    df = pd.read_csv(src)

    iso_onehot = [1.0 if code == AISI5140_ISO_GROUP else 0.0 for code in _ISO_CODES]
    n = len(df)
    X = np.column_stack(
        [
            df["vc_mpm"].to_numpy(dtype=np.float64),
            df["f_mmpr"].to_numpy(dtype=np.float64),
            df["ap_mm"].to_numpy(dtype=np.float64),
            np.full(n, AISI5140_NOSE_RADIUS_MM),
            np.full(n, AISI5140_RAKE_ANGLE_DEG),
            np.tile(iso_onehot, (n, 1)),
            np.full(n, AISI5140_HARDNESS_HB),
        ]
    )
    y = np.column_stack(
        [
            df["vb_mm"].to_numpy(dtype=np.float64),
            df["ra_um"].to_numpy(dtype=np.float64),
        ]
    )
    return X, y
