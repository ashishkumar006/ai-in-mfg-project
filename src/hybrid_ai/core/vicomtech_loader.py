"""Loader for the Vicomtech turning tool-wear dataset.

Source (3.3 MB, public, no login)::

    https://raw.githubusercontent.com/Vicomtech/dataset-machine-tool-wear/main/Dataset_Machine_tool_wear.csv
    -> save as ``data/raw/vicomtech_turning.csv``

2,054 one-second segments from 13 turning tools, each row already
featurised (MEAN/RMS/MAX/SK/KURT per signal plus AE/count features)
with a measured flank-wear label ``Vb`` in **micrometres**.
Typical usage::

    from hybrid_ai.core.vicomtech_loader import load_vicomtech
    X, y, groups = load_vicomtech(split="train")
    # X: (N, F) float32, y: (N, 2) [VB_mm, Ra_um], groups: tool id

Splits are by tool id (train: 9 tools, val: 2, test: 2), so no tool
contributes windows to two splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VICOMTECH_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "vicomtech_turning.csv"

LABEL_COL = "Vb"
GROUP_COL = "tool"

VICOMTECH_TRAIN_TOOLS = (3, 4, 5, 6, 7, 9, 10, 11, 12)
VICOMTECH_VAL_TOOLS = (13, 14)
VICOMTECH_TEST_TOOLS = (15, 16)


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Vicomtech dataset not found at {path}. Download it from "
            "https://raw.githubusercontent.com/Vicomtech/dataset-machine-tool-wear/main/"
            "Dataset_Machine_tool_wear.csv"
        )
    return path


def feature_columns(path: str | Path | None = None) -> list[str]:
    """Return the numeric feature columns (everything but tool/Vb)."""
    import pandas as pd

    src = Path(path) if path is not None else _check_file(VICOMTECH_PATH)
    df = pd.read_csv(src, nrows=0)
    return [c for c in df.columns if c not in (LABEL_COL, GROUP_COL)]


def load_vicomtech(
    path: str | Path | None = None,
    *,
    split: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load featurised turning segments and return ``(X, y, groups)``.

    Parameters
    ----------
    split:
        ``"all"``, ``"train"`` (9 tools), ``"val"`` (2 tools) or
        ``"test"`` (tools 15 + 16).

    Returns
    -------
    X:
        ``(N, F)`` float32 feature matrix (NaNs median-imputed).
    y:
        ``(N, 2)`` float32 ``[VB_mm, Ra_um]``; Vb is converted from µm
        to mm, Ra is a physics proxy from programmed feed/rev and VB.
    groups:
        ``(N,)`` int tool id per row.
    """
    if split not in {"all", "train", "val", "test"}:
        raise ValueError(f"split must be all|train|val|test, got {split!r}")
    src = Path(path) if path is not None else _check_file(VICOMTECH_PATH)
    import pandas as pd

    df = pd.read_csv(src)
    feats = [c for c in df.columns if c not in (LABEL_COL, GROUP_COL)]
    X = df[feats].to_numpy(dtype=np.float64)
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = np.isnan(X)
    X[missing] = np.take(medians, np.where(missing)[1])

    vb_mm = df[LABEL_COL].to_numpy(dtype=np.float64) / 1000.0
    vb_mm = np.nan_to_num(vb_mm, nan=0.0, posinf=0.0, neginf=0.0)
    feed_rev = _feed_per_rev(df)
    ra = np.array([_impute_ra_turning(f, v) for f, v in zip(feed_rev, vb_mm)])
    y = np.stack([np.clip(vb_mm, 0.0, 2.0), ra], axis=1).astype(np.float32)
    groups = df[GROUP_COL].to_numpy(dtype=np.int64)

    if split == "train":
        mask = np.isin(groups, list(VICOMTECH_TRAIN_TOOLS))
    elif split == "val":
        mask = np.isin(groups, list(VICOMTECH_VAL_TOOLS))
    elif split == "test":
        mask = np.isin(groups, list(VICOMTECH_TEST_TOOLS))
    else:
        return X.astype(np.float32), y, groups
    return X[mask].astype(np.float32), y[mask], groups[mask]


def _feed_per_rev(df) -> np.ndarray:
    """Programmed feed per revolution from CNC channels (mm/rev)."""
    try:
        feed = df["CON.G.FREAL"].to_numpy(dtype=np.float64)
        rpm = df["CON.A.SREAL.S"].to_numpy(dtype=np.float64)
        with np.errstate(all="ignore"):
            out = np.where(rpm > 1e-9, feed / np.maximum(rpm, 1e-9), np.nan)
        return np.nan_to_num(out, nan=0.3, posinf=0.3, neginf=0.3)
    except KeyError:
        return np.full(len(df), 0.3)


def _impute_ra_turning(feed_rev: float, vb_mm: float) -> float:
    base = 1.15 * (max(feed_rev, 0.02) / 0.3) ** 0.7
    return float(max(0.25, min(5.5, base + 0.55 * max(vb_mm, 0.0))))
