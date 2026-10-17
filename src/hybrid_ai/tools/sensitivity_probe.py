"""Tool: one-at-a-time sensitivity probe around an operating point.

Diagnoses the centre point plus/minus ``rel_delta`` on each axis
(Vc, feed, depth) via :mod:`hybrid_ai.tools.diagnose_state`, so the
agent can see which direction flips the fuzzy state and which axis
moves wear/finish fastest. Read-only; every probe is a normal
diagnose call with clamped, in-envelope parameters.
"""

from __future__ import annotations

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import diagnose_state
from hybrid_ai.tools.schemas import PART_SCHEMA, POINT_SCHEMA

_BOUNDS = {
    "v_c_mpm": (10.0, 1000.0),
    "f_mmpr": (0.01, 2.0),
    "ap_mm": (0.1, 10.0),
}

TOOL_SPEC: dict = {
    "name": "sensitivity_probe",
    "description": ("Probe model sensitivity: diagnose the centre point "
                    "and ±rel_delta perturbations on each cutting axis. "
                    "Read-only; flags which axis flips the safe/caution/"
                    "critical state and ranks axes by wear/finish movement."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "point": POINT_SCHEMA,
            "insert_code": {"type": ["string", "null"]},
            "sensor_source": {"type": "string", "enum": ["synthetic", "measured"],
                              "default": "synthetic"},
            "rel_delta": {"type": "number", "minimum": 0.01, "maximum": 0.5,
                          "default": 0.1},
        },
        "required": ["part", "point"],
    },
}


def _clamp(axis: str, value: float) -> float:
    """Clamp a perturbed value to the valid parameter bounds."""
    lo, hi = _BOUNDS[axis]
    return min(hi, max(lo, value))


def run(part: PartSpec, point: OperatingPoint,
        insert_code: str | None = None,
        sensor_source: str = "synthetic",
        rel_delta: float = 0.1) -> dict:
    """Probe ±rel_delta on each axis and rank sensitivity."""
    delta = float(rel_delta)
    base = diagnose_state.run(part, point, insert_code=insert_code,
                              sensor_source=sensor_source)
    centre = {"v_b_mm": base["v_b_mm"], "ra_um": base["ra_um"],
              "state": base["state"]}
    probes = []
    movement: dict[str, float] = {}
    for axis in ("v_c_mpm", "f_mmpr", "ap_mm"):
        axis_move = 0.0
        for sign in (-1.0, 1.0):
            raw = getattr(point, axis) * (1.0 + sign * delta)
            pert = OperatingPoint(
                v_c_mpm=_clamp("v_c_mpm", raw) if axis == "v_c_mpm"
                else point.v_c_mpm,
                f_mmpr=_clamp("f_mmpr", raw) if axis == "f_mmpr"
                else point.f_mmpr,
                ap_mm=_clamp("ap_mm", raw) if axis == "ap_mm" else point.ap_mm,
            )
            diag = diagnose_state.run(part, pert, insert_code=base["tool"],
                                      sensor_source=sensor_source)
            dv = diag["v_b_mm"] - centre["v_b_mm"]
            dr = diag["ra_um"] - centre["ra_um"]
            axis_move += abs(dv) + abs(dr)
            probes.append({
                "axis": axis,
                "direction": "+" if sign > 0 else "-",
                "point": pert.to_dict(),
                "v_b_mm": diag["v_b_mm"],
                "ra_um": diag["ra_um"],
                "state": diag["state"],
                "delta_v_b_mm": round(dv, 4),
                "delta_ra_um": round(dr, 4),
                "state_changed": diag["state"] != centre["state"],
            })
        movement[axis] = round(axis_move, 4)
    ranking = sorted(movement, key=lambda a: movement[a], reverse=True)
    return {
        "tool": base["tool"],
        "centre_point": point.to_dict(),
        "centre": centre,
        "rel_delta": delta,
        "probes": probes,
        "axis_movement": movement,
        "most_sensitive_axis": ranking[0],
        "any_state_flip": any(p["state_changed"] for p in probes),
        "mutated": False,
    }
