"""Unified multi-condition window pool: PHM 2010 + NUAA orthogonal.

Both bundles are resampled to 25 Hz and reduced to the three shared
channels ``[force_z, vibration_x, vibration_y]`` so one architecture
can train across materials, machines and cutting conditions::

    X: (N, 3, 256) float32, y: (N, 2) [VB_mm, Ra_um], groups: "src:exp"

Splits are by experiment (never by row), so overlapping windows
cannot leak across them:

- train: PHM c1, c4 + NUAA W1–W6
- val: NUAA W7, W8 (early stopping)
- test: PHM c6 + NUAA W9 (in-family and out-of-family)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hybrid_ai.core.phm_loader import PHM_CHANNELS, PHM_PATH

NUAA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "nuaa_highres.csv"
UNIFIED_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "unified_windows.npz"
)

TARGET_FS_HZ = 25.0
SHARED_CHANNELS = ["force_z", "vibration_x", "vibration_y"]

TRAIN_GROUPS = ("phm:c1", "phm:c4", "nuaa:W1", "nuaa:W2", "nuaa:W3",
                "nuaa:W4", "nuaa:W5", "nuaa:W6")
VAL_GROUPS = ("nuaa:W7", "nuaa:W8")
TEST_GROUPS = ("phm:c6", "nuaa:W9")


def _resample(df, channels: list[str], wear_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Linear-resample one experiment to the target rate."""
    t = df["timestamp"].to_numpy(dtype=np.float64)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(t, kind="stable")
    t = t[order]
    sig = np.nan_to_num(df[channels].to_numpy(dtype=np.float64)[order],
                        nan=0.0, posinf=0.0, neginf=0.0)
    wear = np.nan_to_num(df[wear_col].to_numpy(dtype=np.float64)[order],
                         nan=0.0, posinf=0.0, neginf=0.0)
    step = 1.0 / TARGET_FS_HZ
    grid = np.arange(t[0], t[-1] + step / 2, step)
    if len(grid) < 2:
        return np.zeros((0, len(channels))), np.zeros((0,))
    out_sig = np.stack([np.interp(grid, t, sig[:, j]) for j in range(sig.shape[1])],
                       axis=1)
    out_wear = np.interp(grid, t, wear)
    return out_sig, np.clip(out_wear, 0.0, 3.0)


def _impute_ra_unified(vb: float) -> float:
    return float(max(0.25, min(5.5, 1.0 + 0.55 * max(vb, 0.0))))


def load_unified_windows(
    *,
    window: int = 256,
    stride: int = 16,
    split: str = "all",
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (or load) the pooled window set and return ``(X, y, groups)``."""
    if split not in {"all", "train", "val", "test"}:
        raise ValueError(f"split must be all|train|val|test, got {split!r}")
    cache_key = f"v1 window={window} stride={stride}"
    if use_cache and UNIFIED_CACHE_PATH.exists():
        try:
            blob = np.load(UNIFIED_CACHE_PATH, allow_pickle=True)
            if str(blob.get("_cache_key", "")) == cache_key:
                X = np.asarray(blob["X"])
                y = np.asarray(blob["y"])
                groups = np.asarray(blob["groups"])
                return _select(X, y, groups, split)
        except Exception:
            pass

    import pandas as pd

    windows: list[np.ndarray] = []
    targets: list[list[float]] = []
    tags: list[str] = []

    phm = pd.read_csv(str(PHM_PATH),
                      usecols=["experiment_tag", "timestamp", "tool_wear",
                               *PHM_CHANNELS])
    for cutter, frame in phm.groupby("experiment_tag", sort=True):
        sig, wear = _resample(frame.rename(columns={"tool_wear": "w"}),
                              PHM_CHANNELS, "w")
        _slice(sig, wear, window, stride, f"phm:{cutter}",
               windows, targets, tags, shared_idx=[2, 3, 4])

    nuaa = pd.read_csv(str(NUAA_PATH),
                       usecols=["experiment_tag", "timestamp", "tool_wear",
                                *SHARED_CHANNELS])
    for exp, frame in nuaa.groupby("experiment_tag", sort=True):
        sig, wear = _resample(frame.rename(columns={"tool_wear": "w"}),
                              SHARED_CHANNELS, "w")
        _slice(sig, wear, window, stride, f"nuaa:{exp}",
               windows, targets, tags, shared_idx=[0, 1, 2])

    X = np.asarray(windows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(tags)
    if use_cache:
        UNIFIED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(UNIFIED_CACHE_PATH, X=X, y=y, groups=groups,
                            _cache_key=np.array(cache_key, dtype=object))
    return _select(X, y, groups, split)


def _slice(sig: np.ndarray, wear: np.ndarray, window: int, stride: int,
           tag: str, windows: list, targets: list, tags: list,
           shared_idx: list[int]) -> None:
    for start in range(0, len(sig) - window + 1, stride):
        chunk = sig[start:start + window][:, shared_idx]
        if not np.all(np.isfinite(chunk)):
            continue
        vb = float(wear[start + window - 1])
        if not np.isfinite(vb) or vb < 0.0:
            continue
        windows.append(chunk.T.astype(np.float32))
        targets.append([vb, _impute_ra_unified(vb)])
        tags.append(tag)


def _select(x: np.ndarray, y: np.ndarray, groups: np.ndarray, split: str
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = {"train": TRAIN_GROUPS, "val": VAL_GROUPS, "test": TEST_GROUPS}
    if split == "all":
        return x, y, groups
    mask = np.isin(groups, list(table[split]))
    return x[mask], y[mask], groups[mask]
