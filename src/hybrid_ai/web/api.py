"""FastAPI backend for the Hybrid AI manufacturing pipeline.

Thin HTTP layer over the existing stages (CLIPS / ANN / Fuzzy /
NSGA-II). Serves the single-page frontend from ``/`` and JSON from
``/api/*``. Run with::

    uvicorn hybrid_ai.web.api:app --port 8502

(``PYTHONPATH`` must include ``src/``.)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hybrid_ai.stages.ann.model import (
    TrainConfig,
    WearRoughnessDL,
    WearRoughnessModel,
    predict_for_point,
)
from hybrid_ai.stages.ann.vicomtech_model import VicomtechTurningModel
from hybrid_ai.stages.clips.rules import build_capp_sheet, build_rules, rank_tools
from hybrid_ai.core.nasa_loader import load_blended_dataset, load_nasa_windows
from hybrid_ai.core.nasa_loader import load_nasa_eval_set, measured_sensor_from_window
from hybrid_ai.core.phm_loader import load_phm_windows
from hybrid_ai.core.vicomtech_loader import load_vicomtech
from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.core.sqlite_loader import ToolCatalogRepository
from hybrid_ai.core.synthetic import (
    build_default_tool_catalog,
    generate_synthetic_dataset,
    generate_synthetic_sensor,
)
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

app = FastAPI(title="Hybrid AI Manufacturing", version="0.2.0")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class PartIn(BaseModel):
    part_id: str = "BRACKET-A1"
    material: str = "AISI 1045 steel"
    iso_group: Literal["P", "M", "K", "N", "S"] = "P"
    operation: Literal["turning", "facing", "milling", "drilling", "boring"] = "turning"
    machine: Literal["light", "medium", "heavy"] = "medium"
    hardness_hb: float = 220.0
    target_ra_um: float = 1.6
    batch_size: int = 50


class PointIn(BaseModel):
    v_c_mpm: float
    f_mmpr: float
    ap_mm: float


class ParetoIn(BaseModel):
    part: PartIn
    insert_code: str | None = None
    pop_size: int = Field(default=60, ge=8, le=200)
    n_generations: int = Field(default=40, ge=3, le=200)
    seed: int = Field(default=42, ge=0)
    safe_only: bool = True


class DiagnoseIn(BaseModel):
    part: PartIn
    point: PointIn
    insert_code: str | None = None
    sensor_source: Literal["synthetic", "measured"] = "synthetic"


# ---------------------------------------------------------------------------
# Shared state (lazy singletons: fast startup, models load on first use)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_mlp: WearRoughnessModel | None = None
_cnn1d: WearRoughnessDL | None = None
_cnn_lstm: WearRoughnessDL | None = None
_big: WearRoughnessDL | None = None
_big_missing: bool = False
_turning: VicomtechTurningModel | None = None
_turning_missing: bool = False
_catalog: list | None = None


def _to_part(p: PartIn) -> PartSpec:
    return PartSpec(
        part_id=p.part_id,
        material=p.material,
        iso_group=p.iso_group,
        operation=p.operation,
        machine=p.machine,
        hardness_hb=p.hardness_hb,
        target_ra_um=p.target_ra_um,
        batch_size=p.batch_size,
    )


def get_catalog() -> list:
    global _catalog
    if _catalog is None:
        with _lock:
            if _catalog is None:
                repo = ToolCatalogRepository(":memory:")
                repo.upsert(build_default_tool_catalog())
                _catalog = repo.all()
    return _catalog


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


def _load_dl(name: str) -> WearRoughnessDL:
    return WearRoughnessDL.load(MODELS_DIR / name)


def get_cnn1d() -> WearRoughnessDL:
    global _cnn1d
    if _cnn1d is None:
        with _lock:
            if _cnn1d is None:
                _cnn1d = _load_dl("nasa_cnn1d.pt")
    return _cnn1d


def get_cnn_lstm() -> WearRoughnessDL:
    global _cnn_lstm
    if _cnn_lstm is None:
        with _lock:
            if _cnn_lstm is None:
                _cnn_lstm = _load_dl("nasa_cnn_lstm.pt")
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
                    except Exception:
                        _big_missing = True
                else:
                    _big_missing = True
    return _big


def _resolve_tool(part: PartSpec, insert_code: str | None):
    catalog = get_catalog()
    if insert_code is not None:
        for tool in catalog:
            if tool.insert_code == insert_code:
                return tool
        raise HTTPException(status_code=404,
                            detail=f"unknown insert_code {insert_code!r}")
    _, candidates, selected, _, _ = build_capp_sheet(part, catalog)
    if selected is None:
        raise HTTPException(status_code=422,
                            detail="no catalog tool matches the part spec")
    return selected


def _synthesise_raw_window(point: OperatingPoint, sensor) -> np.ndarray:
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


def _phm_reference() -> tuple[np.ndarray, float, float]:
    x, y, _ = load_phm_windows(split="test")
    j = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[j], float(y[j, 0]), float(y[j, 1])


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


def _vicomtech_reference() -> tuple[np.ndarray, float, float]:
    x, y, _ = load_vicomtech(split="test")
    j = int(np.argsort(y[:, 0])[len(y) // 2])
    return x[j], float(y[j, 0]), float(y[j, 1])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    return {
        "catalog_size": len(get_catalog()),
        "models": {
            "mlp": _mlp is not None,
            "cnn1d": _cnn1d is not None,
            "cnn_lstm": _cnn_lstm is not None,
            "cnn_bilstm_phm": _big is not None,
        },
        "checkpoints": {
            name: (MODELS_DIR / name).exists()
            for name in ("nasa_mlp.pt", "nasa_cnn1d.pt",
                         "nasa_cnn_lstm.pt", "phm_cnn_bilstm.pt")
        },
        "data": {
            "nasa": (ROOT / "data" / "raw" / "nasa_milling.parquet").exists(),
            "phm": (ROOT / "data" / "raw" / "phm2010_highres.csv").exists(),
        },
    }


@app.get("/api/catalog")
def catalog() -> dict:
    return {"tools": [t.to_dict() for t in get_catalog()]}


@app.post("/api/capp")
def capp(part: PartIn) -> dict:
    spec = _to_part(part)
    catalog = get_catalog()
    operations, candidates, selected, envelope, notes = build_capp_sheet(spec, catalog)
    ranked = rank_tools(spec, candidates)
    body: dict = {
        "operations": operations,
        "candidates": [
            {**t.to_dict(), "score": round(s, 3)} for t, s in ranked
        ],
        "selected": selected.to_dict() if selected else None,
        "envelope": envelope,
        "notes": notes,
    }
    if selected is None:
        rules = build_rules()
        body["no_match"] = {
            "per_rule": [
                {
                    "rule": r.name,
                    "passing": sum(1 for t in catalog if r.predicate(spec, t)),
                    "requires": r.description,
                }
                for r in rules
            ],
            "closest": [
                {
                    "insert_code": t.insert_code,
                    "iso_group": t.iso_group,
                    "operation": t.operation,
                    "score": round(s, 3),
                }
                for t, s in rank_tools(spec, list(catalog))[:3]
            ],
        }
    return body


def _resolve_sensor(body: DiagnoseIn, point: OperatingPoint):
    """Synthetic reading, or a measured one from a real NASA run."""
    if body.sensor_source == "measured":
        eval_set = load_nasa_eval_set()
        order = np.argsort(eval_set["y"][:, 0])
        window = eval_set["windows"][int(order[len(order) // 2])]
        return measured_sensor_from_window(
            window, v_c_mpm=point.v_c_mpm, f_mmpr=point.f_mmpr, ap_mm=point.ap_mm,
        ), "measured NASA run"
    sensor = generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr, point.ap_mm,
                                       seed=7)
    return sensor, "synthetic"


@app.post("/api/compare")
def compare(body: DiagnoseIn) -> dict:
    spec = _to_part(body.part)
    tool = _resolve_tool(spec, body.insert_code)
    point = OperatingPoint(v_c_mpm=body.point.v_c_mpm, f_mmpr=body.point.f_mmpr,
                           ap_mm=body.point.ap_mm)
    sensor, _ = _resolve_sensor(body, point)
    rows: list[dict] = []
    pred = predict_for_point(get_mlp(), tool, point,
                             hardness_hb=spec.hardness_hb, sensor=sensor)
    rows.append({"model": "MLP (blended)", "v_b_mm": round(pred.v_b_mm, 3),
                 "ra_um": round(pred.ra_um, 3),
                 "confidence": round(pred.model_confidence, 2),
                 "uncertainty": round(pred.uncertainty, 4)})
    raw = _synthesise_raw_window(point, sensor)[np.newaxis, :, :]
    for name, getter in (("CNN1D", get_cnn1d), ("CNN-LSTM", get_cnn_lstm)):
        try:
            mean, std = getter().predict_with_uncertainty(raw, n_passes=20)
            out = mean[0]
            rows.append({"model": name,
                         "v_b_mm": round(float(np.clip(out[0], 0.02, 1.5)), 3),
                         "ra_um": round(float(np.clip(out[1], 0.2, 6.0)), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4)})
        except Exception as exc:
            rows.append({"model": name, "v_b_mm": None, "ra_um": None,
                         "confidence": None, "error": str(exc)})
    big = get_big()
    if big is not None:
        try:
            ref, true_vb, _ = _phm_reference()
            mean, std = big.predict_with_uncertainty(ref[np.newaxis], n_passes=20)
            out = mean[0]
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "uncertainty": round(float(std[0][0]), 4),
                         "note": f"reference PHM c6 window, true VB={true_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "CNN-BiLSTM (PHM 6.5k)", "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    turning = get_turning()
    if turning is not None:
        try:
            trow, t_vb, _ = _vicomtech_reference()
            out = turning.predict(trow[np.newaxis])[0]
            rows.append({"model": "HGBR (Vicomtech turning)",
                         "v_b_mm": round(float(out[0]), 3),
                         "ra_um": round(float(out[1]), 3),
                         "confidence": None,
                         "note": f"reference turning segment, true VB={t_vb:.2f}"})
        except Exception as exc:
            rows.append({"model": "HGBR (Vicomtech turning)", "v_b_mm": None,
                         "ra_um": None, "confidence": None, "error": str(exc)})
    return {"point": point.to_dict(), "tool": tool.insert_code, "rows": rows}


@app.post("/api/diagnose")
def diagnose_point(body: DiagnoseIn) -> dict:
    spec = _to_part(body.part)
    tool = _resolve_tool(spec, body.insert_code)
    point = OperatingPoint(v_c_mpm=body.point.v_c_mpm, f_mmpr=body.point.f_mmpr,
                           ap_mm=body.point.ap_mm)
    sensor, sensor_label = _resolve_sensor(body, point)
    pred = predict_for_point(get_mlp(), tool, point,
                             hardness_hb=spec.hardness_hb, sensor=sensor)
    diag = diagnose(pred, sensor, target_ra_um=spec.target_ra_um)
    return {
        "point": point.to_dict(),
        "tool": tool.insert_code,
        "sensor_source": sensor_label,
        "v_b_mm": round(pred.v_b_mm, 3),
        "ra_um": round(pred.ra_um, 3),
        "confidence": round(pred.model_confidence, 2),
        "sensor": {
            "vibration_rms_g": round(sensor.vibration_rms_g, 3),
            "current_a": round(sensor.current_a, 2),
            "acoustic_rms_db": round(sensor.acoustic_rms_db, 1),
            "cutting_force_n": round(sensor.cutting_force_n, 1),
        },
        "state": diag.state,
        "explanation": diag.explanation,
        "memberships": {
            "wear": round(diag.wear_membership, 3),
            "roughness": round(diag.roughness_membership, 3),
            "vibration": round(diag.vibration_membership, 3),
        },
    }


@app.post("/api/pareto")
def pareto(body: ParetoIn) -> dict:
    spec = _to_part(body.part)
    tool = _resolve_tool(spec, body.insert_code)

    def _sensor(point: OperatingPoint):
        return generate_synthetic_sensor(point.v_c_mpm, point.f_mmpr,
                                         point.ap_mm, seed=7)

    solutions = optimise(
        get_mlp(), tool, hardness_hb=spec.hardness_hb,
        config=NsgaConfig(pop_size=body.pop_size,
                          n_generations=body.n_generations, seed=body.seed,
                          enforce_safe=body.safe_only),
        sensor_factory=_sensor,
        target_ra_um=spec.target_ra_um,
    )
    rows = [s.to_dict() for s in solutions]
    if not rows:
        return {"tool": tool.insert_code, "solutions": [], "extremes": {},
                "safe_only": body.safe_only, "note": "no feasible solutions"}
    arr = {k: [r[k] for r in rows]
           for k in ("mrr_cm3_per_min", "tool_life_min", "ra_um")}
    pick = lambda key, i: rows[int(np.argmax(arr[key]) if i == "max" else np.argmin(arr[key]))]
    return {
        "tool": tool.insert_code,
        "safe_only": body.safe_only,
        "solutions": rows,
        "extremes": {
            "max_mrr": pick("mrr_cm3_per_min", "max"),
            "max_life": pick("tool_life_min", "max"),
            "min_ra": pick("ra_um", "min"),
        },
    }


@app.get("/api/model_metrics")
def model_metrics() -> dict:
    rows: list[dict] = []
    mlp = get_mlp()
    try:
        xe, ye = load_blended_dataset(n_synthetic=1200, seed=99)
    except Exception:
        xe, ye = generate_synthetic_dataset(n_samples=300, seed=99)
    m = mlp.evaluate(xe, ye)
    rows.append({"model": "MLP", "input": "11 features (blended)",
                 "rmse": round(m["rmse"], 4), "mae": round(m["mae"], 4),
                 "n": m["n"]})
    try:
        xw, yw = load_nasa_windows()
        src, n, kind = "NASA windows", None, "6×9000 raw windows"
    except Exception:
        rng = np.random.default_rng(99)
        xw = rng.normal(size=(60, 6, 9000))
        yw = np.stack([xw[:, 0].mean(axis=1), xw[:, 1].mean(axis=1)], axis=1)
        src, n, kind = "synthetic", None, "6×9000 raw windows"
    for name, getter in (("CNN1D", get_cnn1d), ("CNN-LSTM", get_cnn_lstm)):
        try:
            e = getter().evaluate(xw, yw)
            rows.append({"model": name, "input": kind,
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"]})
        except Exception as exc:
            rows.append({"model": name, "input": kind, "error": str(exc)})
    big = get_big()
    if big is not None:
        try:
            from hybrid_ai.core.phm_loader import load_phm_windows as _phm

            xp, yp, _ = _phm(split="test")
            e = big.evaluate(xp, yp)
            rows.append({"model": "CNN-BiLSTM", "input": "7×256 PHM (c6)",
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"]})
        except Exception as exc:
            rows.append({"model": "CNN-BiLSTM", "input": "7×256 PHM (c6)",
                         "error": str(exc)})
    turning = get_turning()
    if turning is not None:
        try:
            xv, yv, _ = load_vicomtech(split="test")
            e = turning.evaluate(xv, yv)
            rows.append({"model": "HGBR", "input": "152 feats (Vicomtech tools)",
                         "rmse": round(e["rmse"], 4), "mae": round(e["mae"], 4),
                         "n": e["n"]})
        except Exception as exc:
            rows.append({"model": "HGBR", "input": "152 feats (Vicomtech tools)",
                         "error": str(exc)})
    return {"rows": rows}


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
