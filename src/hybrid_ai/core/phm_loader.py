"""Loader + windower for the PHM 2010 high-resolution bundle.

Source (12 MB, public, no login)::

    https://raw.githubusercontent.com/katulu-io/uniwear-dataset/main/data/phm2010_bundle_high_resolution.csv
    -> save as ``data/raw/phm2010_highres.csv``

Tidy mirror of the PHM Society 2010 Data Challenge milling cutters
c1 / c4 / c6: 104,675 rows at ~25 Hz, 7 sensor channels, per-row
flank-wear labels in mm. Typical usage::

    from hybrid_ai.core.phm_loader import load_phm_windows
    X, y, groups = load_phm_windows()
    # X: (N, 7, 256) float32, y: (N, 2) [VB_mm, Ra_um], groups: cutter id

Windows never cross experiment boundaries. Split by cutter id
(``train`` = c1 + c4, ``test`` = c6) so overlapping windows cannot
leak across the split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PHM_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "phm2010_highres.csv"
PHM_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "phm_windows.npz"

PHM_CHANNELS = [
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
]
PHM_FS_HZ = 25.0
PHM_TRAIN_CUTTERS = ("c1", "c4")
PHM_TEST_CUTTERS = ("c6",)


def _impute_ra_phm(vb: float) -> float:
    """Physics proxy Ra for PHM cuts (constant cutting conditions).

    Mirrors :func:`hybrid_ai.core.nasa_loader._impute_ra` with the
    PHM feed (~0.05 mm/tooth): base finish ~1.0 um, degrading with VB.
    """
    return float(max(0.25, min(5.5, 1.0 + 0.55 * max(vb, 0.0))))


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"PHM 2010 bundle not found at {path}. Download it from "
            "https://raw.githubusercontent.com/katulu-io/uniwear-dataset/main/data/"
            "phm2010_bundle_high_resolution.csv"
        )
    return path


def load_phm_windows(
    path: str | Path | None = None,
    *,
    window: int = 256,
    stride: int = 16,
    split: str = "all",
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows over each cutter and return ``(X, y, groups)``.

    Parameters
    ----------
    window:
        Samples per window (256 @25 Hz ~= 10 s of cutting).
    stride:
        Step between window starts. Overlapping windows are standard
        for CNN training; group labels keep splits honest.
    split:
        ``"all"``, ``"train"`` (c1 + c4) or ``"test"`` (c6).
    use_cache:
        Cache the ``(window, stride)`` build under
        ``data/processed/phm_windows.npz``.

    Returns
    -------
    X:
        ``(N, 7, window)`` float32 sensor windows.
    y:
        ``(N, 2)`` float32 ``[VB_mm, Ra_um]``; the label is the wear at
        the window *end* (causal: predict current wear from past signal).
    groups:
        ``(N,)`` cutter id per window for group-aware splits.
    """
    if split not in {"all", "train", "test"}:
        raise ValueError(f"split must be all|train|test, got {split!r}")
    src = Path(path) if path is not None else _check_file(PHM_PATH)
    cache = PHM_CACHE_PATH

    cache_key = f"v1 window={window} stride={stride}"
    if use_cache and cache.exists():
        try:
            blob = np.load(cache, allow_pickle=True)
            if str(blob.get("_cache_key", "")) == cache_key:
                X = np.asarray(blob["X"])
                y = np.asarray(blob["y"])
                groups = np.asarray(blob["groups"])
                return _select_split(X, y, groups, split)
        except Exception:
            pass

    import pandas as pd

    df = pd.read_csv(src, usecols=["experiment_tag", "tool_wear", *PHM_CHANNELS])

    windows: list[np.ndarray] = []
    targets: list[list[float]] = []
    tags: list[str] = []
    for cutter, frame in df.groupby("experiment_tag", sort=True):
        sig = frame[PHM_CHANNELS].to_numpy(dtype=np.float64)
        wear = frame["tool_wear"].to_numpy(dtype=np.float64)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        for start in range(0, len(frame) - window + 1, stride):
            chunk = sig[start:start + window]
            if not np.all(np.isfinite(chunk)):
                continue
            vb = float(wear[start + window - 1])
            if not np.isfinite(vb) or vb < 0.0:
                continue
            windows.append(chunk.T.astype(np.float32))
            targets.append([vb, _impute_ra_phm(vb)])
            tags.append(str(cutter))

    X = np.asarray(windows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(tags)

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, X=X, y=y, groups=groups,
            _cache_key=np.array(cache_key),
        )
    return _select_split(X, y, groups, split)


def _select_split(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "train":
        mask = np.isin(groups, list(PHM_TRAIN_CUTTERS))
    elif split == "test":
        mask = np.isin(groups, list(PHM_TEST_CUTTERS))
    else:
        return x, y, groups
    return x[mask], y[mask], groups[mask]
