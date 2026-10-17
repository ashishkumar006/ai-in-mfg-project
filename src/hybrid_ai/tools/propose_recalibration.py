"""Tool: proposal-only fuzzy-threshold recalibration analysis (dry-run).

Compares the current :class:`FuzzyThresholds` against candidate
overrides over a wear/roughness grid and reports which cells would
flip state. This tool has no apply path by construction: it never
touches production thresholds, checkpoints, or files. A human applies
an accepted proposal by editing the stage and re-running the suite.
"""

from __future__ import annotations

from hybrid_ai.core.schemas import (
    OperatingPoint,
    SensorReading,
    ToolCandidate,
    WearPrediction,
)
from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds, diagnose

_AXES = (
    "v_b_low", "v_b_mid", "v_b_high",
    "ra_low", "ra_mid", "ra_high",
    "vib_low", "vib_mid", "vib_high",
)

_TOOL = ToolCandidate(
    insert_code="PROBE", iso_group="P", operation="turning",
    nose_radius_mm=0.8, rake_angle_deg=6.0,
    v_c_min_mpm=100.0, v_c_max_mpm=400.0,
    f_min_mmpr=0.1, f_max_mmpr=0.5, ap_max_mm=3.0,
)
_POINT = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.3, ap_mm=1.5)

TOOL_SPEC: dict = {
    "name": "propose_recalibration",
    "description": ("Dry-run recalibration proposal: compare current fuzzy "
                    "thresholds against candidate overrides over a "
                    "wear/roughness grid and report state flips. "
                    "PROPOSAL-ONLY: never applies anything; a human applies "
                    "accepted proposals and re-runs the test suite."),
    "parameters": {
        "type": "object",
        "properties": {
            "target_ra_um": {"type": "number", "minimum": 0.1,
                             "maximum": 10.0, "default": 1.6},
            "vibration_g": {"type": "number", "minimum": 0.0,
                            "maximum": 5.0, "default": 1.0},
            **{axis: {"type": ["number", "null"]} for axis in _AXES},
        },
        "required": [],
    },
}


def _check_order(values: dict[str, float], prefix: str) -> None:
    """Reject candidates that break low < mid < high ordering."""
    from hybrid_ai.tools.errors import ToolError

    lo, mid, hi = (values[f"{prefix}_low"], values[f"{prefix}_mid"],
                   values[f"{prefix}_high"])
    if not (0 < lo < mid < hi):
        raise ToolError("bad_args",
                        f"candidate {prefix} thresholds must satisfy "
                        f"0 < low < mid < high, got {lo}/{mid}/{hi}")


def run(target_ra_um: float = 1.6, vibration_g: float = 1.0,
        v_b_low: float | None = None, v_b_mid: float | None = None,
        v_b_high: float | None = None, ra_low: float | None = None,
        ra_mid: float | None = None, ra_high: float | None = None,
        vib_low: float | None = None, vib_mid: float | None = None,
        vib_high: float | None = None) -> dict:
    """Evaluate candidate thresholds over a grid; apply nothing."""
    from hybrid_ai.tools.errors import ToolError

    overrides = {"v_b_low": v_b_low, "v_b_mid": v_b_mid,
                 "v_b_high": v_b_high, "ra_low": ra_low, "ra_mid": ra_mid,
                 "ra_high": ra_high, "vib_low": vib_low, "vib_mid": vib_mid,
                 "vib_high": vib_high}
    current = FuzzyThresholds.for_target(float(target_ra_um))
    cand_values = {axis: getattr(current, axis) for axis in _AXES}
    changed = []
    for axis in _AXES:
        value = overrides[axis]
        if value is not None:
            cand_values[axis] = float(value)
            if float(value) != getattr(current, axis):
                changed.append(axis)
    for prefix in ("v_b", "ra", "vib"):
        _check_order(cand_values, prefix)
    candidate = FuzzyThresholds(**cand_values)
    sensor = SensorReading(
        vibration_rms_g=float(vibration_g), current_a=5.0,
        acoustic_rms_db=70.0, cutting_force_n=500.0,
    )
    flips: list[dict] = []
    counts = {"safe_to_caution": 0, "relaxes": 0, "to_critical": 0,
              "from_critical": 0, "other": 0}
    for i in range(10):
        v_b = 0.05 + i * 0.05
        for j in range(10):
            ra = 0.4 + j * 0.3
            wear = WearPrediction(tool=_TOOL, point=_POINT,
                                  v_b_mm=v_b, ra_um=ra)
            before = diagnose(wear, sensor, thresholds=current).state
            after = diagnose(wear, sensor, thresholds=candidate).state
            if before == after:
                continue
            key = "other"
            if before == "safe" and after == "caution":
                key = "safe_to_caution"
            elif before == "caution" and after == "safe":
                key = "relaxes"
            elif after == "critical":
                key = "to_critical"
            elif before == "critical":
                key = "from_critical"
            counts[key] += 1
            if len(flips) < 25:
                flips.append({"v_b_mm": round(v_b, 2), "ra_um": round(ra, 2),
                              "before": before, "after": after})
    return {
        "mode": "proposal-only",
        "applied": False,
        "mutated": False,
        "current": {axis: getattr(current, axis) for axis in _AXES},
        "candidate": {axis: getattr(candidate, axis) for axis in _AXES},
        "changed_axes": changed,
        "grid_cells": 100,
        "n_flips": sum(counts.values()),
        "flip_counts": counts,
        "flips": flips,
        "apply_instructions": ("Human only: edit FuzzyThresholds defaults in "
                               "src/hybrid_ai/stages/fuzzy/diagnoser.py, then "
                               "run pytest tests/stages/fuzzy and update "
                               "docs/VALIDATION.md before deploying."),
    }
