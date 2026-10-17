"""Loader for the NASA Milling dataset.

The file is expected at ``data/raw/nasa_milling.parquet`` (HuggingFace
mirror of the original NASA Open Data Portal release). It contains
167 runs with scalar cutting parameters and list-typed sensor
time-series.

Typical usage::

    from hybrid_ai.core.nasa_loader import load_nasa_dataset
    X, y = load_nasa_dataset()
    # X: (n_samples, n_features), y: (n_samples, 2) -> [VB, Ra]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from hybrid_ai.core.schemas import (
    ISOMaterialGroup,
    OperatingPoint,
    SensorReading,
    ToolCandidate,
)

NASA_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "nasa_milling.parquet"
NASA_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "nasa_features.npz"

_MATERIAL_TO_ISO: dict[int, ISOMaterialGroup] = {
    1: "M",
    2: "K",
}

_MATERIAL_HARDNESS_HB: dict[int, float] = {
    1: 180.0,
    2: 220.0,
}

_DEFAULT_V_C_MPM: dict[int, float] = {
    1: 150.0,
    2: 200.0,
}

_SENSOR_COLS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
_WINDOW_LEN = 9000


def _check_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"NASA dataset not found at {path}. "
            "Download it from https://huggingface.co/datasets/jonasmaltebecker/nasa_milling "
            "and save as data/raw/nasa_milling.parquet."
        )
    return path


def _safe_list(arr, *, clip: float = 20.0) -> np.ndarray:
    if isinstance(arr, (list, tuple, np.ndarray)):
        out = np.asarray(arr, dtype=np.float64)
    else:
        out = np.asarray(list(arr), dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out


def _impute_ra(feed: float, vb: float) -> float:
    """Physics-based Ra proxy for NASA runs (no measured Ra column).

    Theoretical roughness scales with feed^0.7; worn tools degrade
    finish roughly linearly with VB. Coefficients chosen so typical
    NASA feeds (0.25/0.5 mm/rev) map to 0.9-2.0 um, inside the fuzzy
    thresholds (0.8/1.6/2.4).
    """
    base = 1.15 * (max(feed, 0.05) / 0.3) ** 0.7
    wear_term = 0.55 * max(vb, 0.0)
    return float(max(0.25, min(5.5, base + wear_term)))


def _truncate(arr: np.ndarray, length: int = _WINDOW_LEN) -> np.ndarray:
    if arr.shape[0] >= length:
        return arr[:length].copy()
    pad = np.zeros(length, dtype=arr.dtype)
    pad[: arr.shape[0]] = arr
    return pad


def _featurise_window(signal: np.ndarray, fs: float = 1.0) -> np.ndarray:
    signal = _truncate(np.asarray(signal, dtype=np.float64))
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    crest = peak / max(rms, 1e-12)
    try:
        kurt = float(stats.kurtosis(signal, fisher=True))
    except Exception:
        kurt = 0.0
    if not np.isfinite(kurt):
        kurt = 0.0
    spec = np.abs(rfft(signal))
    freqs = rfftfreq(signal.shape[0], d=1.0 / max(fs, 1e-12))
    if spec.size == 0:
        peak_freq, spec_centroid, spec_energy = 0.0, 0.0, 0.0
    else:
        total = spec.sum() + 1e-12
        peak_freq = float(freqs[np.argmax(spec)])
        spec_centroid = float(np.sum(freqs * spec) / total)
        spec_energy = float(np.sum(spec ** 2))
    return np.array(
        [rms, peak, crest, kurt, peak_freq, spec_centroid, spec_energy],
        dtype=np.float64,
    )


def _is_corrupted(row: dict, *, thresh: float = 10.0) -> bool:
    """Detect mirror-corrupted records (values >> physical range)."""
    for col in _SENSOR_COLS:
        try:
            arr = np.asarray(row.get(col, []), dtype=np.float64).ravel()
        except Exception:
            continue
        if arr.size == 0:
            continue
        with np.errstate(all="ignore"):
            peak = float(np.nanmax(np.abs(arr[np.isfinite(arr)]))) if np.any(np.isfinite(arr)) else 0.0
        if peak > thresh:
            return True
    return False


def _featurise_sensors(row: dict, fs: float = 1.0) -> np.ndarray:
    feats: list[np.ndarray] = []
    for col in _SENSOR_COLS:
        arr = _safe_list(row.get(col, []))
        feats.append(_featurise_window(arr, fs=fs))
    return np.concatenate(feats)


def load_nasa_dataset(
    path: str | Path | None = None,
    *,
    fs: float = 1.0,
    use_cache: bool = True,
    drop_nan_vb: bool = True,
    impute_ra: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the NASA milling dataset and return ``(X, y)`` arrays.

    Parameters
    ----------
    path:
        Location of the parquet file. Defaults to
        ``data/raw/nasa_milling.parquet``.
    fs:
        Assumed sampling frequency (Hz) for spectral features.
        The original NASA dataset does not publish fs; default 1.0
        keeps features unit-agnostic. Set to the real value if known.
    use_cache:
        If True, write/read a cached ``.npz`` under
        ``data/processed/nasa_features.npz`` keyed by the input
        ``fs`` so repeated calls skip feature extraction.
    drop_nan_vb:
        Drop runs where VB is missing (21 of 167).
    impute_ra:
        If True (default), fill Ra with a physics-based proxy from feed
        and VB since NASA has no measured Ra column. If False, Ra is 0.
    Returns
    -------
    X:
        ``(n_samples, n_features)`` float64 array. Features are:
        [V_c, feed, DOC, hardness_hb, iso_P, iso_M, iso_K, iso_N, iso_S,
         sensor_features...]
    y:
        ``(n_samples, 2)`` float64 array. Columns are [VB_mm, Ra_um].
    """
    src = Path(path) if path is not None else _check_file(NASA_PATH)
    cache = NASA_CACHE_PATH

    cache_key = f"v2 fs={fs} impute_ra={impute_ra}"
    if use_cache and cache.exists():
        try:
            blob = np.load(cache, allow_pickle=False)
            if str(blob.get("_cache_key", "")) == cache_key:
                X = blob["X"]
                y = blob["y"]
                if drop_nan_vb:
                    mask = ~np.isnan(y[:, 0])
                    X, y = X[mask], y[mask]
                return X, y
        except Exception:
            pass

    import pandas as pd

    df = pd.read_parquet(src)

    if drop_nan_vb:
        df = df.dropna(subset=["VB"]).reset_index(drop=True)

    rows: list[np.ndarray] = []
    targets: list[list[float]] = []

    for _, row in df.iterrows():
        rd = row.to_dict()
        if _is_corrupted(rd):
            continue
        material_code = int(row["material"])
        iso_group = _MATERIAL_TO_ISO.get(material_code, "P")
        hardness = _MATERIAL_HARDNESS_HB.get(material_code, 200.0)
        v_c = _DEFAULT_V_C_MPM.get(material_code, 180.0)

        iso_codes: list[ISOMaterialGroup] = ["P", "M", "K", "N", "S"]
        iso_onehot = [1.0 if code == iso_group else 0.0 for code in iso_codes]

        sensor_feats = _featurise_sensors(rd, fs=fs)
        feature_vec = np.concatenate(
            [
                np.array([v_c, float(row["feed"]), float(row["DOC"]), hardness], dtype=np.float64),
                np.array(iso_onehot, dtype=np.float64),
                sensor_feats,
            ]
        )
        rows.append(feature_vec)
        vb = float(row["VB"]) if not np.isnan(row["VB"]) else 0.0
        ra = _impute_ra(float(row["feed"]), vb) if impute_ra else 0.0
        targets.append([vb, ra])

    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            X=X,
            y=y,
            _cache_key=np.array(cache_key, dtype=object),
        )

    return X, y


def load_nasa_operating_points(
    path: str | Path | None = None,
) -> tuple[list[OperatingPoint], list[ToolCandidate], np.ndarray]:
    """Return operating points, a synthetic tool envelope, and VB labels.

    The NASA dataset records cutting conditions but not insert geometry.
    This helper builds a generic ``ToolCandidate`` per material and
    returns the corresponding operating points so the ANN can be
    trained/evaluated without a real catalog.
    """
    src = Path(path) if path is not None else _check_file(NASA_PATH)
    import pandas as pd

    df = pd.read_parquet(src)
    if df["VB"].isna().any():
        df = df.dropna(subset=["VB"]).reset_index(drop=True)

    points: list[OperatingPoint] = []
    tools: list[ToolCandidate] = []
    labels: list[float] = []

    for _, row in df.iterrows():
        material_code = int(row["material"])
        iso_group = _MATERIAL_TO_ISO.get(material_code, "P")
        v_c = _DEFAULT_V_C_MPM.get(material_code, 180.0)
        feed = float(row["feed"])
        doc = float(row["DOC"])

        tool = ToolCandidate(
            insert_code=f"NASA-{material_code}",
            iso_group=iso_group,
            operation="milling",
            nose_radius_mm=0.8,
            rake_angle_deg=0.0,
            v_c_min_mpm=max(50.0, v_c * 0.7),
            v_c_max_mpm=min(500.0, v_c * 1.3),
            f_min_mmpr=max(0.05, feed * 0.7),
            f_max_mmpr=min(1.0, feed * 1.3),
            ap_max_mm=max(1.0, doc * 1.3),
            description=f"Generic insert for NASA material {material_code}.",
        )
        points.append(OperatingPoint(v_c_mpm=v_c, f_mmpr=feed, ap_mm=doc))
        tools.append(tool)
        labels.append(float(row["VB"]))

    return points, tools, np.asarray(labels, dtype=np.float64)


def nasa_feature_dim(fs: float = 1.0) -> int:
    """Return the expected feature dimension for the NASA loader.

    4 scalar params + 5 ISO one-hot + 6 channels × 7 features/channel.
    """
    return 4 + 5 + len(_SENSOR_COLS) * 7


def load_nasa_eval_set(
    path: str | Path | None = None,
    *,
    window_len: int = 9000,
) -> dict:
    """Aligned NASA eval samples for model-showdown demos.

    Iterates the parquet file once (NaN-VB and corrupted rows dropped)
    and returns windows, operating points, generic tools and targets
    in one consistent order so MLP (feature) and CNN (window) models
    can be compared on the exact same run.
    """
    src = Path(path) if path is not None else _check_file(NASA_PATH)
    import pandas as pd

    df = pd.read_parquet(src)
    df = df.dropna(subset=["VB"]).reset_index(drop=True)

    windows: list[np.ndarray] = []
    points: list[OperatingPoint] = []
    tools: list[ToolCandidate] = []
    labels: list[list[float]] = []

    for _, row in df.iterrows():
        rd = row.to_dict()
        if _is_corrupted(rd):
            continue
        channels = [_truncate(_safe_list(rd.get(col, [])), window_len)
                    for col in _SENSOR_COLS]
        if any(c.shape[0] != window_len for c in channels):
            continue
        material_code = int(row["material"])
        iso_group = _MATERIAL_TO_ISO.get(material_code, "P")
        hardness = _MATERIAL_HARDNESS_HB.get(material_code, 200.0)
        v_c = _DEFAULT_V_C_MPM.get(material_code, 180.0)
        feed, doc = float(row["feed"]), float(row["DOC"])
        vb = float(row["VB"])
        windows.append(np.stack(channels))
        points.append(OperatingPoint(v_c_mpm=v_c, f_mmpr=feed, ap_mm=doc))
        tools.append(ToolCandidate(
            insert_code=f"NASA-{material_code}", iso_group=iso_group,
            operation="milling", nose_radius_mm=0.8, rake_angle_deg=0.0,
            v_c_min_mpm=max(50.0, v_c * 0.7),
            v_c_max_mpm=min(500.0, v_c * 1.3),
            f_min_mmpr=max(0.05, feed * 0.7),
            f_max_mmpr=min(1.0, feed * 1.3),
            ap_max_mm=max(1.0, doc * 1.3),
            description=f"Generic insert for NASA material {material_code}.",
        ))
        labels.append([vb, _impute_ra(feed, vb)])
    return {
        "windows": np.asarray(windows, dtype=np.float64),
        "points": points,
        "tools": tools,
        "y": np.asarray(labels, dtype=np.float64),
        "hardness_hb": hardness,
    }


def measured_sensor_from_window(
    window: np.ndarray,
    *,
    v_c_mpm: float,
    f_mmpr: float,
    ap_mm: float,
) -> SensorReading:
    """Build a sensor reading from a real measured NASA window.

    Channels follow ``_SENSOR_COLS`` order. Vibration is the worst of
    the table/spindle RMS (conservative for safety diagnosis);
    current is the DC-spindle mean; acoustic emission converts V to
    dB via the 20 µPa reference. NASA records no cutting force, so
    force is estimated from the standard formula and documented as
    such in the acoustic/force provenance.
    """
    chans = [np.asarray(np.ravel(row), dtype=np.float64) for row in window]
    while len(chans) < 6:
        chans.append(np.zeros(1))
    rms = lambda a: float(np.sqrt(np.mean(np.clip(a, -50.0, 50.0) ** 2)))
    vib = max(rms(chans[2]), rms(chans[3]))
    current = float(abs(np.mean(np.clip(chans[1], -50.0, 50.0))))
    ae_rms = max(rms(chans[4]), rms(chans[5]), 1e-9)
    acoustic_db = 20.0 * float(np.log10(ae_rms / 20e-6))
    force = 250.0 * ap_mm * f_mmpr + 80.0 * (v_c_mpm / 200.0)
    return SensorReading(
        vibration_rms_g=vib,
        current_a=current,
        acoustic_rms_db=float(np.clip(acoustic_db, 30.0, 120.0)),
        cutting_force_n=float(max(20.0, force)),
        spindle_speed_rpm=float(max(500.0, 1000.0 * v_c_mpm / max(0.1, 3.14159))),
    )
    """Return the expected feature dimension for the NASA loader.

    4 scalar params + 5 ISO one-hot + 6 channels × 7 features/channel.
    """
    return 4 + 5 + len(_SENSOR_COLS) * 7


def load_nasa_synthetic_features(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map NASA runs onto the 11-dim synthetic feature schema.

    Columns: ``[V_c, feed, ap, nose_radius, rake_angle, iso×5, hardness]``
    with a generic ``nose=0.8 mm, rake=0 deg`` insert. VB is measured;
    Ra uses :func:`_impute_ra`. Lets one MLP train on blended
    synthetic + real data without a schema break.
    """
    X51, y = load_nasa_dataset(path=path)
    n = X51.shape[0]
    X11 = np.column_stack(
        [
            X51[:, 0],
            X51[:, 1],
            X51[:, 2],
            np.full(n, 0.8),
            np.zeros(n),
            X51[:, 4:9],
            X51[:, 3],
        ]
    )
    return X11, y


def load_nasa_labeled_features(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NASA rows in the 11-dim schema, with their run and case ids.

    Same features as :func:`load_nasa_synthetic_features` (so a blended
    model can consume them) but keeps the ``run`` / ``case`` labels so a
    cut can be held out honestly — by cutter/run, not by random row,
    which is the split that actually measures generalisation.

    Returns
    -------
    X11, y, runs, cases :
        Feature matrix, ``[VB, Ra]`` targets, and the per-row run and
        case ids. Row order matches the parquet (minus NaN-VB and
        corrupted rows), so labels stay aligned.
    """
    import pandas as pd

    src = Path(path) if path is not None else _check_file(NASA_PATH)
    df = pd.read_parquet(src).dropna(subset=["VB"]).reset_index(drop=True)

    iso_codes: list[ISOMaterialGroup] = ["P", "M", "K", "N", "S"]
    X: list[list[float]] = []
    y: list[list[float]] = []
    runs: list[int] = []
    cases: list[int] = []

    for _, row in df.iterrows():
        rd = row.to_dict()
        if _is_corrupted(rd):
            continue
        material_code = int(row["material"])
        iso_group = _MATERIAL_TO_ISO.get(material_code, "P")
        hardness = _MATERIAL_HARDNESS_HB.get(material_code, 200.0)
        v_c = _DEFAULT_V_C_MPM.get(material_code, 180.0)
        iso_onehot = [1.0 if code == iso_group else 0.0 for code in iso_codes]
        feed = float(row["feed"])
        vb = float(row["VB"])
        X.append([v_c, feed, float(row["DOC"]), 0.8, 0.0, *iso_onehot, hardness])
        y.append([vb, _impute_ra(feed, vb)])
        runs.append(int(row["run"]))
        cases.append(int(row["case"]))

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(runs, dtype=np.int64),
        np.asarray(cases, dtype=np.int64),
    )


def load_blended_dataset(
    n_synthetic: int = 1200,
    seed: int = 42,
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (wide envelope) + NASA-mapped (real) rows, shuffled."""
    from hybrid_ai.core.synthetic import generate_synthetic_dataset

    Xs, ys = generate_synthetic_dataset(n_samples=n_synthetic, seed=seed)
    try:
        Xn, yn = load_nasa_synthetic_features(path=path)
        X = np.concatenate([Xs, Xn], axis=0)
        y = np.concatenate([ys, yn], axis=0)
    except FileNotFoundError:
        X, y = Xs, ys
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


def load_nasa_windows(
    path: str | Path | None = None,
    *,
    fs: float = 1.0,
    window_len: int = 9000,
    drop_nan_vb: bool = True,
    impute_ra: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw sensor windows and VB labels for CNN / CNN-LSTM training.

    Each sample is ``(n_channels, window_len)`` with the six sensor
    channels stacked as rows. ``n_channels=6``. Signals are clipped to
    ``[-20, 20]`` to suppress single corrupted records in the public
    mirror.

    Parameters
    ----------
    path:
        Parquet file path. Defaults to ``data/raw/nasa_milling.parquet``.
    fs:
        Sampling frequency (Hz). Currently unused but kept for API
        symmetry with the featurised loader.
    window_len:
        Truncation / padding length. Default 9000 matches the NASA
        record length.
    drop_nan_vb:
        Drop runs with missing VB measurements.
    impute_ra:
        Fill Ra with the same physics proxy as :func:`load_nasa_dataset`.

    Returns
    -------
    X:
        ``(n_samples, n_channels, window_len)`` float64 array.
    y:
        ``(n_samples, 2)`` float64 array. ``[VB_mm, Ra_um]``.
    """
    src = Path(path) if path is not None else _check_file(NASA_PATH)
    import pandas as pd

    df = pd.read_parquet(src)
    if drop_nan_vb:
        df = df.dropna(subset=["VB"]).reset_index(drop=True)

    windows: list[np.ndarray] = []
    targets: list[list[float]] = []

    for _, row in df.iterrows():
        if _is_corrupted(row.to_dict()):
            continue
        channels = [_truncate(_safe_list(row.get(col, [])), window_len) for col in _SENSOR_COLS]
        if any(c.shape[0] != window_len for c in channels):
            continue
        stack = np.stack(channels)
        windows.append(stack)
        vb = float(row["VB"]) if not np.isnan(row["VB"]) else 0.0
        ra = _impute_ra(float(row["feed"]), vb) if impute_ra else 0.0
        targets.append([vb, ra])

    X = np.asarray(windows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    return X, y
