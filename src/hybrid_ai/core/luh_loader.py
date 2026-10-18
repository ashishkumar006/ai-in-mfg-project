"""Loader for the LUH/KIT milling tool-wear dataset (Denkena et al. 2023).

Source: Mendeley Data DOI ``10.17632/zpxs87bjt8`` (CC BY); published in
Data in Brief, doi ``10.1016/j.dib.2023.109574``. Nine end-milling
cutters (three per machine, three machines) worn from fresh to
VB ~ 150 um in shoulder milling of cast iron 600-3/S; one HDF5 file
per run (6,418 total).

This repo keeps the full dataset (6,418 h5 files, ~10.6 GB) under
``data/raw/luh_milling/h5/``; a lighter every-4th-run mirror (~2.6 GB)
is enough for smoke tests — ``scripts/checks/download_luh_subset.py``
fetches both (pass ``every=1`` for the full set). The featuriser
consumes whatever h5 files are present.

Signals per run (~4.4 s):

- dynamometer forces ``force_sensor_{x,y,z}`` at 25 kHz (Kistler,
  10 kHz low-pass);
- machine controls at 500 Hz: ``torque_spindle``, position control
  deviations x/y.

Typical usage::

    from hybrid_ai.core.luh_loader import load_luh_features
    X, vb_mm, contact_s, tools, runs = load_luh_features()
    # X: (n, 16) float64 signal features; vb_mm: wear label (mm)

Caveats (from the dataset paper): wear labels are measured every 1-2
layers (~40 runs) and linearly interpolated for other runs (+-5 um
microscope uncertainty); tools T4-T6 have aliasing in machine-side
force/torque signals (spindle torque included here); tools T7-T8 were
run with a rotated workpiece coordinate system (affects machine-side
position signals). Exposure time C is the true cumulative tool contact
time at run start (seconds).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LUH_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "luh_milling"
LUH_H5_DIR = LUH_DIR / "h5"
LUH_FILELIST = LUH_DIR / "filelist.csv"
LUH_CACHE = Path(__file__).resolve().parents[3] / "data" / "processed" / "luh_features.npz"

LUH_FS_SENSOR_HZ = 25_000.0
LUH_FORCE_CLIP_N = 3000.0

LUH_TOOL_MACHINE: dict[int, int] = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
}

FEATURE_NAMES: tuple[str, ...] = (
    "force_x_rms_n",
    "force_y_rms_n",
    "force_z_rms_n",
    "mag_mean_n",
    "mag_std_n",
    "mag_kurt",
    "mag_skew",
    "mag_band_0_500_log",
    "mag_band_500_2k_log",
    "mag_band_2k_6k_log",
    "mag_band_6k_10k_log",
    "torque_spindle_mean",
    "torque_spindle_std",
    "torque_spindle_kurt",
    "control_dev_x_std",
    "control_dev_y_std",
)

_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 500.0),
    (500.0, 2000.0),
    (2000.0, 6000.0),
    (6000.0, 10000.0),
)


def _check_inputs(h5_dir: Path, filelist: Path) -> tuple[Path, Path]:
    if not filelist.exists():
        raise FileNotFoundError(
            f"LUH filelist not found at {filelist}. Download "
            "'filelist.csv' from https://data.mendeley.com/datasets/zpxs87bjt8 "
            "(see scripts/checks/download_luh_subset.py)."
        )
    if not h5_dir.exists() or not any(h5_dir.glob("*.h5")):
        raise FileNotFoundError(
            f"No LUH h5 files under {h5_dir}. Run "
            "scripts/checks/download_luh_subset.py to fetch the subset."
        )
    return h5_dir, filelist


def _stats(arr: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    kurt = float(stats.kurtosis(arr, fisher=True))
    skew = float(stats.skew(arr))
    return (kurt if np.isfinite(kurt) else 0.0, skew if np.isfinite(skew) else 0.0)


def featurise_file(path: str | Path) -> np.ndarray:
    """Extract the 16 signal features from one run's HDF5 file.

    Parameters
    ----------
    path:
        HDF5 file for a single run.

    Returns
    -------
    np.ndarray
        ``(16,)`` float64 feature vector in ``FEATURE_NAMES`` order.
    """
    import h5py

    with h5py.File(path, "r") as h:
        fx = np.asarray(h["signals_sensor/force_sensor_x"][0], dtype=np.float64)
        fy = np.asarray(h["signals_sensor/force_sensor_y"][0], dtype=np.float64)
        fz = np.asarray(h["signals_sensor/force_sensor_z"][0], dtype=np.float64)
        torque = np.asarray(h["signals_machine/torque_spindle"][0], dtype=np.float64)
        pcd_x = np.asarray(
            h["signals_machine/position_control_deviation_axis_x"][0], dtype=np.float64
        )
        pcd_y = np.asarray(
            h["signals_machine/position_control_deviation_axis_y"][0], dtype=np.float64
        )

    clip = LUH_FORCE_CLIP_N
    fx = np.clip(np.nan_to_num(fx), -clip, clip)
    fy = np.clip(np.nan_to_num(fy), -clip, clip)
    fz = np.clip(np.nan_to_num(fz), -clip, clip)
    torque = np.nan_to_num(torque)
    pcd_x = np.nan_to_num(pcd_x)
    pcd_y = np.nan_to_num(pcd_y)

    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
    mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    mag_kurt, mag_skew = _stats(mag)

    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs = np.fft.rfftfreq(mag.size, d=1.0 / LUH_FS_SENSOR_HZ)
    power = spec ** 2
    bands = [
        float(np.log10(power[(freqs >= lo) & (freqs < hi)].sum() + 1.0))
        for lo, hi in _BANDS_HZ
    ]

    torque_kurt, _ = _stats(torque)
    return np.array(
        [
            rms(fx), rms(fy), rms(fz),
            float(mag.mean()), float(mag.std()), mag_kurt, mag_skew,
            *bands,
            float(torque.mean()), float(torque.std()), torque_kurt,
            float(pcd_x.std()), float(pcd_y.std()),
        ],
        dtype=np.float64,
    )


def _filelist_rows(h5_dir: Path, filelist: Path) -> list[dict]:
    import csv

    rows: list[dict] = []
    with open(filelist, "r", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if (h5_dir / r["filename"]).exists():
                rows.append(r)
    rows.sort(key=lambda r: r["filename"])
    return rows


def load_luh_features(
    *,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Featurise every run present locally and return aligned arrays.

    Parameters
    ----------
    use_cache:
        If True, cache features under ``data/processed/luh_features.npz``
        keyed by the file set, so repeat calls skip the h5 reads.

    Returns
    -------
    X:
        ``(n, 16)`` float64 features (``FEATURE_NAMES``).
    vb_mm:
        ``(n,)`` wear label in mm (measured/interpolated ``wear`` um).
    contact_s:
        ``(n,)`` cumulative tool contact time at run start (s).
    tools, runs:
        ``(n,)`` tool and run ids (9 tools; ``LUH_TOOL_MACHINE`` maps
        tools to machines).
    """
    import hashlib

    h5_dir, filelist = _check_inputs(LUH_H5_DIR, LUH_FILELIST)
    rows = _filelist_rows(h5_dir, filelist)

    names = [r["filename"] for r in rows]
    cache_key = "v1 n=%d sha=%s" % (
        len(names),
        hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:16],
    )

    if use_cache and LUH_CACHE.exists():
        try:
            blob = np.load(LUH_CACHE, allow_pickle=False)
            if str(blob.get("_cache_key", "")) == cache_key:
                return (
                    blob["X"], blob["vb_mm"], blob["contact_s"],
                    blob["tools"], blob["runs"],
                )
        except Exception:
            pass

    X = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    vb_mm = np.empty(len(rows), dtype=np.float64)
    contact_s = np.empty(len(rows), dtype=np.float64)
    tools = np.empty(len(rows), dtype=np.int64)
    runs = np.empty(len(rows), dtype=np.int64)

    for i, row in enumerate(rows):
        X[i] = featurise_file(h5_dir / row["filename"])
        vb_mm[i] = float(row["wear"]) / 1000.0
        contact_s[i] = float(row["cumulated_tool_contact_time"])
        tools[i] = int(row["tool"])
        runs[i] = int(row["run"])

    if use_cache:
        LUH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            LUH_CACHE,
            X=X, vb_mm=vb_mm, contact_s=contact_s, tools=tools, runs=runs,
            _cache_key=np.array(cache_key),
        )

    return X, vb_mm, contact_s, tools, runs
