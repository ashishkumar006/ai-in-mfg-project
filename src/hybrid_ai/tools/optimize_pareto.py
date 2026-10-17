"""Tool: multi-objective cutting-parameter optimisation (NSGA-II stage)."""

from __future__ import annotations

import numpy as np

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise, tool_life_min
from hybrid_ai.stages.nsga2.taylor import compare_life
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA

try:  # pymoo ships an exact 3-objective hypervolume indicator
    from pymoo.indicators.hv import HV

    _HAS_HV = True
except Exception:  # pragma: no cover - optional dep path
    _HAS_HV = False

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
                "hypervolume": None, "spread": None, "taylor_check": None,
                "no_safe_front": _explain_no_front(part, tool, safe_only)}
    arr = {k: [r[k] for r in rows]
           for k in ("mrr_cm3_per_min", "tool_life_min", "ra_um")}
    pick = lambda key, i: rows[int(np.argmax(arr[key]) if i == "max" else np.argmin(arr[key]))]
    extremes = {
        "max_mrr": pick("mrr_cm3_per_min", "max"),
        "max_life": pick("tool_life_min", "max"),
        "min_ra": pick("ra_um", "min"),
    }
    return {
        "tool": tool.insert_code,
        "safe_only": safe_only,
        "solutions": rows,
        "extremes": extremes,
        "hypervolume": _hypervolume(arr),
        "spread": _front_spread(arr),
        "taylor_check": _taylor_check(part, tool, extremes),
    }


def _hypervolume(arr: dict) -> dict | None:
    """3-objective hypervolume of the front (pymoo, exact).

    Objective vector is the optimiser's minimised form
    ``[-MRR, -tool_life, R_a]``. The reference point is derived from the
    front itself (10% worse than the best point on each axis) so the
    number is comparable *between runs of the same problem* rather than
    an absolute physical quantity.
    """
    if not _HAS_HV or not arr.get("mrr_cm3_per_min"):
        return None
    f = np.column_stack([
        -np.asarray(arr["mrr_cm3_per_min"], dtype=float),
        -np.asarray(arr["tool_life_min"], dtype=float),
        np.asarray(arr["ra_um"], dtype=float),
    ])
    lo, hi = f.min(axis=0), f.max(axis=0)
    spread = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    ref = hi + 0.1 * spread
    hv = HV(ref_point=ref)(f)
    return {
        "value": float(hv),
        "n_points": int(f.shape[0]),
        "ref_point": [round(float(v), 4) for v in ref],
        "method": "pymoo HV (exact, 3 objectives)",
    }


def _front_spread(arr: dict) -> dict | None:
    """Spread of the front in optimiser objective space.

    Two numbers, both on the minimised objective matrix
    ``[-MRR, -tool_life, R_a]`` scaled to the unit cube per axis:

    - ``spacing``: coefficient of variation of the nearest-neighbour
      distances (Schott's spacing idea). 0 means the returned points sit
      on a perfectly uniform grid; large means the front is clumped.
    - ``extent``: the axis-wise spans of the front, i.e. how much of
      each objective range the trade-off actually covers.

    The deck's validation row names hypervolume *and* spread; this is
    the spread half, independent of the reference-point choice that
    hypervolume depends on.
    """
    pts = arr.get("mrr_cm3_per_min") or []
    if len(pts) < 2:
        return None
    f = np.column_stack([
        -np.asarray(arr["mrr_cm3_per_min"], dtype=float),
        -np.asarray(arr["tool_life_min"], dtype=float),
        np.asarray(arr["ra_um"], dtype=float),
    ])
    lo, hi = f.min(axis=0), f.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    norm = (f - lo) / span
    d = np.linalg.norm(norm[:, None, :] - norm[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    spacing = float(nn.std(ddof=0) / max(nn.mean(), 1e-12))
    return {
        "spacing": spacing,
        "n_points": int(f.shape[0]),
        "extent": [round(float(v), 4) for v in (hi - lo)],
        "method": "Schott spacing (CV of NN distances) on normalised "
                  "objective space; extent = axis spans",
    }


def _taylor_check(part: PartSpec, tool, extremes: dict) -> dict:
    """Compare the ANN's life estimates against the extended Taylor equation.

    The optimiser derives life from predicted flank wear (``6.0 / V_B``).
    Taylor's equation is an independent physical model, so agreement
    between the two is evidence the ANN's wear scale is sane — and
    disagreement is worth surfacing rather than hiding.
    """
    from hybrid_ai.stages.ann.model import predict_for_point

    out: dict = {}
    for name, row in extremes.items():
        point = OperatingPoint(
            v_c_mpm=row["v_c_mpm"], f_mmpr=row["f_mmpr"], ap_mm=row["ap_mm"],
        )
        sensor = _models.sensor_for_pareto(point)
        pred = predict_for_point(_models.get_mlp(), tool, point,
                                 hardness_hb=part.hardness_hb, sensor=sensor)
        ann_life = tool_life_min(point, pred.v_b_mm)
        out[name] = compare_life(pred.v_b_mm, ann_life, point, part.iso_group)
    verdicts = [v["verdict"] for v in out.values()]
    out["summary"] = (
        "consistent" if verdicts.count("consistent") >= 2
        else verdicts[0] if len(set(verdicts)) == 1 else "mixed"
    )
    return out


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
        if diag.state == "safe":  # including the first candidate
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
