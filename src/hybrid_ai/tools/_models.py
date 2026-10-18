"""Shared model/catalog loading for tools (singletons, lazy, thread-safe).

Both dashboards used to duplicate this logic; tools are now the single
home for it. No Streamlit / FastAPI imports here — pure Python plus
functools caching, so CLIs and tests get the same behavior.
"""

from __future__ import annotations

import functools
import threading
from pathlib import Path

import numpy as np

from hybrid_ai.core.nasa_loader import (
    load_blended_dataset,
    load_nasa_eval_set,
    measured_sensor_from_window,
)
from hybrid_ai.core.schemas import OperatingPoint
from hybrid_ai.core.sqlite_loader import ToolCatalogRepository
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
)
from hybrid_ai.stages.ann.model import (
    TrainConfig,
    WearRoughnessDL,
    WearRoughnessModel,
)
from hybrid_ai.stages.ann.luh_model import LUHMillingModel
from hybrid_ai.stages.ann.vicomtech_model import VicomtechTurningModel
from hybrid_ai.tools.errors import ToolError

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

_lock = threading.Lock()
_mlp: WearRoughnessModel | None = None
_cnn1d: WearRoughnessDL | None = None
_cnn_lstm: WearRoughnessDL | None = None
_big: WearRoughnessDL | None = None
_big_missing: bool = False
_turning: VicomtechTurningModel | None = None
_turning_missing: bool = False
_luh: LUHMillingModel | None = None
_luh_missing: bool = False
_catalog: list | None = None
_catalog_source: dict | None = None
CATALOG_DB_PATH = ROOT / "data" / "processed" / "tool_catalog.sqlite"


def get_catalog() -> list:
    """The tooling catalog, parsed from the ISO 13399-style CSV and
    served from the ISO-keyed SQLite store.

    Persists the parsed catalog at ``data/processed/tool_catalog.sqlite``
    so the SQLite schema is a real artifact downstream of the parse
    step (CLIPS selection reads it through the repository). Falls back
    to the 15-insert built-in list only if the catalog asset cannot be
    parsed (missing/broken file).
    """
    global _catalog, _catalog_source
    if _catalog is None:
        with _lock:
            if _catalog is None:
                try:
                    from hybrid_ai.core.tooling_catalog import \
                        load_default_iso13399_catalog
                    rows = load_default_iso13399_catalog()
                    source = {"source": "iso13399_csv",
                              "rows": len(rows)}
                    db = CATALOG_DB_PATH
                except Exception:
                    rows = build_default_tool_catalog()
                    source = {"source": "builtin_fallback",
                              "rows": len(rows)}
                    db = ":memory:"
                if db != ":memory:":
                    CATALOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                repo = ToolCatalogRepository(db)
                repo.replace(rows)
                _catalog = repo.all()
                _catalog_source = source
    return _catalog


def catalog_source() -> dict:
    """Provenance of the catalog ``get_catalog()`` serves."""
    if get_catalog() and _catalog_source is not None:
        return dict(_catalog_source)
    return {"source": "unknown", "rows": 0}


def get_mlp() -> WearRoughnessModel:
    global _mlp
    if _mlp is None:
        with _lock:
            if _mlp is None:
                model = WearRoughnessModel()
                ckpt = MODELS_DIR / "nasa_mlp.pt"
                if ckpt.exists():
                    try:
                        model.load(ckpt)
                    except Exception:
                        model = _train_mlp()
                else:
                    model = _train_mlp()
                model._source = "blended synthetic + NASA"  # type: ignore[attr-defined]
                _mlp = model
    return _mlp


def _train_mlp() -> WearRoughnessModel:
    try:
        x, y = load_blended_dataset(n_synthetic=1200, seed=42)
    except Exception:
        x, y = generate_synthetic_dataset(n_samples=1200, seed=42)
    model = WearRoughnessModel()
    model.fit(x, y, TrainConfig(epochs=60))
    return model


def get_cnn1d() -> WearRoughnessDL:
    global _cnn1d
    if _cnn1d is None:
        with _lock:
            if _cnn1d is None:
                _cnn1d = WearRoughnessDL.load(MODELS_DIR / "nasa_cnn1d.pt")
                _cnn1d._source = "NASA windows"  # type: ignore[attr-defined]
    return _cnn1d


def get_cnn_lstm() -> WearRoughnessDL:
    global _cnn_lstm
    if _cnn_lstm is None:
        with _lock:
            if _cnn_lstm is None:
                _cnn_lstm = WearRoughnessDL.load(MODELS_DIR / "nasa_cnn_lstm.pt")
                _cnn_lstm._source = "NASA windows"  # type: ignore[attr-defined]
    return _cnn_lstm


def get_big() -> WearRoughnessDL | None:
    global _big, _big_missing
    if _big is None and not _big_missing:
        with _lock:
            if _big is None and not _big_missing:
                ckpt = MODELS_DIR / "phm_cnn_bilstm.pt"
                if ckpt.exists():
                    try:
                        _big = WearRoughnessDL.load(ckpt)
                        _big._source = "PHM c1+c4"  # type: ignore[attr-defined]
                    except Exception:
                        _big_missing = True
                else:
                    _big_missing = True
    return _big


def get_turning() -> VicomtechTurningModel | None:
    global _turning, _turning_missing
    if _turning is None and not _turning_missing:
        with _lock:
            if _turning is None and not _turning_missing:
                ckpt = MODELS_DIR / "vicomtech_hgbr.joblib"
                if ckpt.exists():
                    try:
                        _turning = VicomtechTurningModel.load(ckpt)
                    except Exception:
                        _turning_missing = True
                else:
                    _turning_missing = True
    return _turning


def get_luh() -> LUHMillingModel | None:
    global _luh, _luh_missing
    if _luh is None and not _luh_missing:
        with _lock:
            if _luh is None and not _luh_missing:
                ckpt = MODELS_DIR / "luh_hgbr.joblib"
                if ckpt.exists():
                    try:
                        _luh = LUHMillingModel.load(ckpt)
                    except Exception:
                        _luh_missing = True
                else:
                    _luh_missing = True
    return _luh


def resolve_tool(part, insert_code: str | None):
    """Return the requested insert, or the top-ranked CAPP pick.

    Raises
    ------
    ToolError
        ``unknown_tool`` or ``no_match``.
    """
    from hybrid_ai.stages.clips.rules import build_capp_sheet

    catalog = get_catalog()
    if insert_code is not None:
        for tool in catalog:
            if tool.insert_code == insert_code:
                return tool
        raise ToolError("unknown_tool", f"unknown insert_code {insert_code!r}")
    _, _, selected, _, _ = build_capp_sheet(part, catalog)
    if selected is None:
        raise ToolError("no_match", "no catalog tool matches the part spec")
    return selected


def resolve_sensor(point: OperatingPoint, sensor_source: str = "synthetic"):
    """Synthetic reading, or a measured one from a real NASA run."""
    if sensor_source == "measured":
        eval_set = load_nasa_eval_set()
        order = np.argsort(eval_set["y"][:, 0])
        window = eval_set["windows"][int(order[len(order) // 2])]
        return measured_sensor_from_window(
            window, v_c_mpm=point.v_c_mpm, f_mmpr=point.f_mmpr,
            ap_mm=point.ap_mm,
        ), "measured NASA run"
    if sensor_source != "synthetic":
        raise ToolError("bad_sensor_source",
                        f"sensor_source must be synthetic|measured, got {sensor_source!r}")
    return generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr, point.ap_mm,
                                     seed=7), "synthetic"


def synthesise_raw_window(point: OperatingPoint, sensor) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.normal(0, sensor.vibration_rms_g, size=9000)
    return np.stack(
        [
            base * 0.9,
            base * 1.0,
            base * 0.8,
            base * (sensor.current_a / 5.0),
            base * (sensor.acoustic_rms_db / 60.0),
            base * (sensor.cutting_force_n / 500.0),
        ]
    ).astype(np.float64)


def sensor_for_pareto(point: OperatingPoint):
    return generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr, point.ap_mm,
                                     seed=7)


@functools.lru_cache(maxsize=1)
def phm_reference() -> tuple[np.ndarray, float, float]:
    """Median-wear window from held-out PHM cutter c6 + true labels."""
    from hybrid_ai.core.phm_loader import load_phm_windows

    x, y, _ = load_phm_windows(split="test")
    idx = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[idx], float(y[idx, 0]), float(y[idx, 1])


@functools.lru_cache(maxsize=1)
def vicomtech_reference() -> tuple[np.ndarray, float, float]:
    """Median-wear segment from held-out Vicomtech tools + true labels."""
    from hybrid_ai.core.vicomtech_loader import load_vicomtech

    x, y, _ = load_vicomtech(split="test")
    idx = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[idx], float(y[idx, 0]), float(y[idx, 1])


@functools.lru_cache(maxsize=1)
def luh_reference() -> tuple[np.ndarray, float]:
    """Median-wear LUH run's assembled feature row + true V_B.

    The shipped checkpoint is refit on all nine tools after the 9-tool
    leave-one-tool-out evaluation, so this reference row may be
    in-sample; the honest metric is the LOTO number in
    ``docs/luh_loto.json``.
    """
    from hybrid_ai.core.luh_loader import load_luh_features

    x16, vb_mm, contact_s, _tools, _runs = load_luh_features()
    idx = int(np.argsort(vb_mm)[len(vb_mm) // 2])
    row = LUHMillingModel.assemble(x16[idx:idx + 1], contact_s[idx:idx + 1])[0]
    return row, float(vb_mm[idx])
