"""Loader for the Haas VF-1 milling RUL dataset (Piecuch & Zabinski 2025).

Source (1.7 MB features + metadata, public, no login)::

    https://springernature.figshare.com/ — article 28589216
    -> FeatureAndMetadata_Milling.csv -> ``data/raw/haas_features.csv``

968 milling cycles from 14 cutting tools (42CrMo4 steel) run to
failure. Each row is one cycle with precomputed vibration/current
features. The label is **cycles remaining** (``CycleToFailure``),
so this feeds remaining-useful-life regression, not VB prediction.
Typical usage::

    from hybrid_ai.core.haas_loader import load_haas
    X, y, groups = load_haas(split="train")
    # X: (N, F) float64, y: (N,) cycles left, groups: tool id

Splits are by tool id (train: 10 tools, val: 2, test: 2 longest
lives), so no tool contributes cycles to two splits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAAS_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "haas_features.csv"

GROUP_COL = "TollIndex"
LABEL_COL = "CycleToFailure"
DROP_COLS = ("FileName", GROUP_COL, LABEL_COL, "CycleToFailureNormalized")

HAAS_TRAIN_TOOLS = (2, 3, 4, 5, 6, 10, 11, 101, 102, 103)
HAAS_VAL_TOOLS = (7, 8)
HAAS_TEST_TOOLS = (9, 105)


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Haas dataset not found at {path}. Download "
            "FeatureAndMetadata_Milling.csv from "
            "https://doi.org/10.6084/m9.figshare.28589216"
        )
    return path


def _read_haas(src: Path, nrows: int | None = None):
    """Read the semicolon CSV, normalising comma decimals to dots."""
    import pandas as pd

    df = pd.read_csv(src, sep=";", skiprows=[0], nrows=nrows)
    for col in df.columns:
        if col != "FileName" and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce")
    return df


def feature_columns(path: str | Path | None = None) -> list[str]:
    """Return the numeric feature columns (drops ids, filename, labels)."""
    src = Path(path) if path is not None else _check_file(HAAS_PATH)
    df = _read_haas(src, nrows=5)
    import pandas as pd

    return [c for c in df.columns
            if c not in DROP_COLS and c != "FileName"
            and pd.api.types.is_numeric_dtype(df[c])]


def load_haas(
    path: str | Path | None = None,
    *,
    split: str = "all",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-cycle features and return ``(X, y, groups)``.

    Parameters
    ----------
    split:
        ``"all"``, ``"train"`` (10 tools), ``"val"`` (tools 7+8) or
        ``"test"`` (tools 9+105, the two longest lives).

    Returns
    -------
    X:
        ``(N, F)`` float64 feature matrix (NaNs median-imputed).
    y:
        ``(N,)`` float64 cycles remaining until failure.
    groups:
        ``(N,)`` int tool id per cycle.
    """
    if split not in {"all", "train", "val", "test"}:
        raise ValueError(f"split must be all|train|val|test, got {split!r}")
    src = Path(path) if path is not None else _check_file(HAAS_PATH)
    import pandas as pd

    df = _read_haas(src)
    feats = [c for c in df.columns if c not in DROP_COLS and c != "FileName"]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].to_numpy(dtype=np.float64)
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = np.isnan(X)
    X[missing] = np.take(medians, np.where(missing)[1])

    y = df[LABEL_COL].to_numpy(dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, 500.0)
    groups = df[GROUP_COL].to_numpy(dtype=np.int64)

    table = {"train": HAAS_TRAIN_TOOLS, "val": HAAS_VAL_TOOLS,
             "test": HAAS_TEST_TOOLS}
    if split == "all":
        return X, y, groups
    mask = np.isin(groups, list(table[split]))
    return X[mask], y[mask], groups[mask]


def load_haas_sequences(
    path: str | Path | None = None,
    *,
    split: str = "all",
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Per-tool ``(X, y, tool_id)`` histories sorted by cycle number.

    Same splits as :func:`load_haas`; rows within each tool follow
    ``NumberOfCycle`` so sequence models see the wear trajectory.
    """
    src = Path(path) if path is not None else _check_file(HAAS_PATH)
    import pandas as pd

    df = _read_haas(src)
    order = df.sort_values([GROUP_COL, "NumberOfCycle"]).reset_index(drop=True)
    table = {"train": HAAS_TRAIN_TOOLS, "val": HAAS_VAL_TOOLS,
             "test": HAAS_TEST_TOOLS}
    if split != "all":
        order = order[order[GROUP_COL].isin(list(table[split]))].reset_index(drop=True)
    feats = [c for c in order.columns if c not in DROP_COLS and c != "FileName"]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(order[c])]
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for tool, frame in order.groupby(GROUP_COL, sort=True):
        X = np.nan_to_num(frame[feats].to_numpy(dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(np.nan_to_num(frame[LABEL_COL].to_numpy(dtype=np.float64)),
                    0.0, 500.0)
        out.append((X, y, int(tool)))
    return out
