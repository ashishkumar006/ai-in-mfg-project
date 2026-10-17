"""FastAPI backend for the Hybrid AI manufacturing pipeline.

Thin HTTP layer: every endpoint delegates to ``hybrid_ai.tools``,
which owns loading, validation and the stage calls. Run with::

    uvicorn hybrid_ai.web.api:app --port 8502

(``PYTHONPATH`` must include ``src/``.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import (
    catalog as catalog_tool,
)
from hybrid_ai.tools import (
    compare_models,
    diagnose_state,
    optimize_pareto,
    propose_recalibration,
    select_tool,
    sensitivity_probe,
    sweep_pareto,
)
from hybrid_ai.tools import _models as tool_models
from hybrid_ai.tools.errors import ToolError

ROOT = Path(__file__).resolve().parents[3]

app = FastAPI(title="Hybrid AI Manufacturing", version="0.4.0")


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


def _to_point(p: PointIn) -> OperatingPoint:
    return OperatingPoint(v_c_mpm=p.v_c_mpm, f_mmpr=p.f_mmpr, ap_mm=p.ap_mm)


def _call(fn, *args, **kwargs):
    """Run a tool, translating ToolError into HTTP status codes."""
    try:
        return fn(*args, **kwargs)
    except ToolError as exc:
        raise HTTPException(
            status_code=404 if exc.code == "unknown_tool" else 422,
            detail=exc.message,
        )


# ---------------------------------------------------------------------------
# Routes (all delegate to hybrid_ai.tools)
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    models_dir = ROOT / "models"
    return {
        "catalog_size": len(tool_models.get_catalog()),
        "models": {
            "mlp": tool_models._mlp is not None,
            "cnn1d": tool_models._cnn1d is not None,
            "cnn_lstm": tool_models._cnn_lstm is not None,
            "cnn_bilstm_phm": tool_models._big is not None,
            "turning": tool_models._turning is not None,
        },
        "checkpoints": {
            name: (models_dir / name).exists()
            for name in ("nasa_mlp.pt", "nasa_cnn1d.pt",
                         "nasa_cnn_lstm.pt", "phm_cnn_bilstm.pt",
                         "vicomtech_hgbr.joblib")
        },
        "data": {
            "nasa": (ROOT / "data" / "raw" / "nasa_milling.parquet").exists(),
            "phm": (ROOT / "data" / "raw" / "phm2010_highres.csv").exists(),
        },
    }


@app.get("/api/catalog")
def catalog() -> dict:
    return catalog_tool.run_list()


@app.post("/api/capp")
def capp(part: PartIn) -> dict:
    return select_tool.run(_to_part(part))


@app.post("/api/compare")
def compare(body: DiagnoseIn) -> dict:
    return _call(
        compare_models.run,
        _to_part(body.part), _to_point(body.point),
        insert_code=body.insert_code, sensor_source=body.sensor_source,
    )


@app.post("/api/diagnose")
def diagnose_point(body: DiagnoseIn) -> dict:
    return _call(
        diagnose_state.run,
        _to_part(body.part), _to_point(body.point),
        insert_code=body.insert_code, sensor_source=body.sensor_source,
    )


@app.post("/api/pareto")
def pareto(body: ParetoIn) -> dict:
    return _call(
        optimize_pareto.run,
        _to_part(body.part), insert_code=body.insert_code,
        pop_size=body.pop_size, n_generations=body.n_generations,
        seed=body.seed, safe_only=body.safe_only,
    )


@app.get("/api/model_metrics")
def model_metrics() -> dict:
    return catalog_tool.run_benchmarks()


class SweepIn(BaseModel):
    part: PartIn
    insert_code: str | None = None
    n_seeds: int = Field(default=3, ge=1, le=5)
    seed_base: int = 42
    pop_size: int = Field(default=60, ge=8, le=200)
    n_generations: int = Field(default=40, ge=3, le=200)
    safe_only: bool = True


class SensitivityIn(BaseModel):
    part: PartIn
    point: PointIn
    insert_code: str | None = None
    sensor_source: Literal["synthetic", "measured"] = "synthetic"
    rel_delta: float = Field(default=0.1, ge=0.01, le=0.5)


class RecalibrateIn(BaseModel):
    target_ra_um: float = Field(default=1.6, ge=0.1, le=10.0)
    vibration_g: float = Field(default=1.0, ge=0.0, le=5.0)
    v_b_low: float | None = None
    v_b_mid: float | None = None
    v_b_high: float | None = None
    ra_low: float | None = None
    ra_mid: float | None = None
    ra_high: float | None = None
    vib_low: float | None = None
    vib_mid: float | None = None
    vib_high: float | None = None


@app.post("/api/sweep")
def sweep(body: SweepIn) -> dict:
    return _call(
        sweep_pareto.run,
        _to_part(body.part), insert_code=body.insert_code,
        n_seeds=body.n_seeds, seed_base=body.seed_base,
        pop_size=body.pop_size, n_generations=body.n_generations,
        safe_only=body.safe_only,
    )


@app.post("/api/sensitivity")
def sensitivity(body: SensitivityIn) -> dict:
    return _call(
        sensitivity_probe.run,
        _to_part(body.part), _to_point(body.point),
        insert_code=body.insert_code, sensor_source=body.sensor_source,
        rel_delta=body.rel_delta,
    )


@app.post("/api/propose_recalibration")
def propose_recalibration_route(body: RecalibrateIn) -> dict:
    return _call(
        propose_recalibration.run,
        target_ra_um=body.target_ra_um, vibration_g=body.vibration_g,
        v_b_low=body.v_b_low, v_b_mid=body.v_b_mid,
        v_b_high=body.v_b_high, ra_low=body.ra_low, ra_mid=body.ra_mid,
        ra_high=body.ra_high, vib_low=body.vib_low, vib_mid=body.vib_mid,
        vib_high=body.vib_high,
    )


_ARCHIVED_AGENT_NOTE = (
    "Agentic endpoints (/api/chat, /api/agent_run, "
    "/api/agent_run_stream) archived to "
    "archive/agentic-setup/web/api.agent_endpoints.py."
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
