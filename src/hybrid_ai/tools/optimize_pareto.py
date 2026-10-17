"""Tool: multi-objective cutting-parameter optimisation (NSGA-II stage)."""

from __future__ import annotations

import numpy as np

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA

TOOL_SPEC: dict = {
    "name": "optimize_pareto",
    "description": ("Search speed/feed/depth for the non-dominated trade-off "
                    "between removal rate, tool life and finish. With "
                    "safe_only (default), every returned point is fuzzy-safe."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "insert_code": {"type": ["string", "null"]},
            "pop_size": {"type": "integer", "minimum": 8, "maximum": 200,
                         "default": 60},
            "n_generations": {"type": "integer", "minimum": 3, "maximum": 200,
                              "default": 40},
            "seed": {"type": "integer", "default": 42},
            "safe_only": {"type": "boolean", "default": True},
        },
        "required": ["part"],
    },
}


def run(part: PartSpec, insert_code: str | None = None,
        pop_size: int = 60, n_generations: int = 40, seed: int = 42,
        safe_only: bool = True) -> dict:
    """Optimise cutting parameters and return front + extremes."""
    tool = _models.resolve_tool(part, insert_code)
    solutions = optimise(
        _models.get_mlp(), tool, hardness_hb=part.hardness_hb,
        config=NsgaConfig(pop_size=int(pop_size),
                          n_generations=int(n_generations), seed=int(seed),
                          enforce_safe=safe_only),
        sensor_factory=_models.sensor_for_pareto,
        target_ra_um=part.target_ra_um,
    )
    rows = [s.to_dict() for s in solutions]
    if not rows:
        return {"tool": tool.insert_code, "safe_only": safe_only,
                "solutions": [], "extremes": {},
                "no_safe_front": _explain_no_front(part, tool, safe_only)}
    arr = {k: [r[k] for r in rows]
           for k in ("mrr_cm3_per_min", "tool_life_min", "ra_um")}
    pick = lambda key, i: rows[int(np.argmax(arr[key]) if i == "max" else np.argmin(arr[key]))]
    return {
        "tool": tool.insert_code,
        "safe_only": safe_only,
        "solutions": rows,
        "extremes": {
            "max_mrr": pick("mrr_cm3_per_min", "max"),
            "max_life": pick("tool_life_min", "max"),
            "min_ra": pick("ra_um", "min"),
        },
    }


def _explain_no_front(part: PartSpec, tool, safe_only: bool) -> dict:
    """Say *why* the search found nothing, using the stage that vetoed it.

    An empty front is a legitimate answer — some material/finish
    combinations have no safe setting at all. Reporting the vetoing
    diagnosis keeps that distinguishable from a search that failed.
    """
    from hybrid_ai.stages.ann.model import predict_for_point
    from hybrid_ai.stages.clips.rules import build_capp_sheet
    from hybrid_ai.stages.fuzzy.diagnoser import diagnose

    _, _, _, envelope, _ = build_capp_sheet(part, _models.get_catalog())
    lo, hi = envelope["v_c_mpm"]
    flo, fhi = envelope["f_mmpr"]
    alo, ahi = envelope["ap_mm"]
    candidates: list[tuple] = [
        (v, f, a) for v in (lo, (lo + hi) / 2, hi)
        for f in (flo, (flo + fhi) / 2, fhi)
        for a in (alo, (alo + ahi) / 2, ahi)
    ]
    best = None
    for v, f, a in candidates:
        point = OperatingPoint(v_c_mpm=v, f_mmpr=f, ap_mm=a)
        sensor = _models.sensor_for_pareto(point)
        pred = predict_for_point(_models.get_mlp(), tool, point,
                                 hardness_hb=part.hardness_hb, sensor=sensor)
        diag = diagnose(pred, sensor, target_ra_um=part.target_ra_um)
        if best is None:
            best = (point, pred, diag)
        elif diag.state == "safe":
            return {"reason": "search_missed_safe_point",
                    "note": ("the gate found no safe point, but one exists in "
                             "the envelope — rerun with a larger population"),
                    "point": point.to_dict(), "diagnosis": _diag_dict(diag, pred)}
    point, pred, diag = best
    out = {
        "reason": "no_safe_setting_exists",
        "note": ("the fuzzy gate rejects every point in this envelope — "
                 "this material/finish combination has no safe setting"),
        "point": point.to_dict(),
        "diagnosis": _diag_dict(diag, pred),
        "envelope": {"v_c_mpm": [lo, hi], "f_mmpr": [flo, fhi],
                     "ap_mm": [alo, ahi]},
        "target_ra_um": part.target_ra_um,
    }
    if safe_only:
        out["hint"] = ("relax the target finish, or turn off the safe-only gate "
                       "to see the full trade-off space (every point will be "
                       "unsafe)")
    return out


def _diag_dict(diag, pred) -> dict:
    return {"state": diag.state, "explanation": diag.explanation,
            "v_b_mm": round(pred.v_b_mm, 3), "ra_um": round(pred.ra_um, 3)}
