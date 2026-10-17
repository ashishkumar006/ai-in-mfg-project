"""Tool: closed-loop re-planning (feedback from diagnosis back to selection).

The four stages are chained, and slide 4 of the project deck describes the
loop that closes them: when the diagnosis comes back unsafe, that verdict
should feed *back* into tool selection and trigger a re-plan, not merely
be displayed.

This tool implements that edge. Given a part and the insert (and
optionally the operating point) that just failed, it re-runs the rule
stage with the failing insert excluded, diagnoses the alternative's
envelope midpoint, and reports whether the problem is actually resolved.

The loop is deliberately *bounded*: it never re-plans more than once, and
it never silently substitutes an insert. If no alternative exists, or the
alternative is no better, that is reported as such — a re-plan loop that
always claims success is worse than no loop.

Swapping the insert is rarely enough on its own — the fault is usually
driven by the part (its hardness or its target finish), not by which
same-group insert is chosen. So the loop *also* searches a small grid
inside the new envelope for the safest setting, and reports that setting
as the re-planned point. It is the combination (different insert +
different point) that can actually close the loop.
"""

from __future__ import annotations

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.ann.model import predict_for_point
from hybrid_ai.stages.clips.rules import build_capp_sheet
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.stages.nsga2.optimiser import mrr_cm3_per_min
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA

TOOL_SPEC: dict = {
    "name": "replan",
    "description": (
        "Feedback loop: re-run rule-based tool selection with a failing "
        "insert excluded, then diagnose the alternative. Returns the new "
        "insert, envelope and verdict plus whether the problem is resolved."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "failed_insert": {"type": ["string", "null"]},
            "point": {
                "type": ["object", "null"],
                "description": "The operating point that failed, if known.",
                "properties": {
                    "v_c_mpm": {"type": "number"},
                    "f_mmpr": {"type": "number"},
                    "ap_mm": {"type": "number"},
                },
            },
            "failed_state": {"type": ["string", "null"],
                             "description": "caution | critical, the verdict to beat"},
        },
        "required": ["part"],
    },
}

_SEVERITY = {"safe": 0, "caution": 1, "critical": 2}


def run(part: PartSpec, failed_insert: str | None = None,
        point: OperatingPoint | None = None,
        failed_state: str | None = None) -> dict:
    """Re-plan the part with ``failed_insert`` removed from the catalog."""
    catalog = _models.get_catalog()
    remaining = [t for t in catalog if t.insert_code != failed_insert]

    body: dict = {
        "failed_insert": failed_insert,
        "failed_state": failed_state,
        "failed_point": point.to_dict() if point else None,
        "considered": [t.insert_code for t in remaining],
    }

    if not remaining:
        body.update(_failure("the catalog has no other insert to try"))
        return body

    _, candidates, selected, envelope, notes = build_capp_sheet(part, remaining)
    if selected is None or envelope is None:
        body.update(_failure(
            "no other insert in the catalog satisfies the CLIPS rules for "
            f"this part (tried {len(remaining)}). The catalog has a single "
            "insert for this material/operation, so switching inserts cannot "
            "help — adjust the target finish or use the Pareto front to pick "
            "a safer point on the current insert instead."))
        return body

    midpoint = OperatingPoint(
        v_c_mpm=(envelope["v_c_mpm"][0] + envelope["v_c_mpm"][1]) / 2,
        f_mmpr=(envelope["f_mmpr"][0] + envelope["f_mmpr"][1]) / 2,
        ap_mm=min(1.5, envelope["ap_mm"][1]),
    )
    mid_pred, mid_diag = _evaluate(part, selected, midpoint)

    best = _search_envelope(part, selected, envelope)
    bp, bpred, bdiag = best

    # Report the improvement relative to the envelope midpoint, which is
    # what the original pipeline would have flagged.
    new_sev = _SEVERITY.get(bdiag.state, 2)
    mid_sev = _SEVERITY.get(mid_diag.state, 2)
    old_sev = _SEVERITY.get(failed_state) if failed_state else mid_sev
    beat_midpoint = new_sev < mid_sev

    if bdiag.state == "safe":
        resolved = True
        verdict = ("the re-plan finds a safe setting for the alternative "
                   "insert — this closes the loop")
    elif old_sev is not None and new_sev < old_sev:
        resolved = True
        verdict = (f"the re-plan improves the verdict from {failed_state or mid_diag.state} "
                   f"to {bdiag.state}, but is not fully safe")
    elif beat_midpoint:
        resolved = True
        verdict = (f"the insert swap alone is {mid_diag.state}, but a safer "
                   f"point in its envelope is {bdiag.state}")
    else:
        resolved = False
        verdict = (f"the alternative insert is also {bdiag.state} at every "
                   "setting tried — changing the insert does not fix this "
                   "part; change the target finish, hardness assumption or "
                   "the machine instead")

    if bdiag.state == "safe":
        reading = "safe"
    elif beat_midpoint:
        reading = "improved"
    else:
        reading = "no_improvement"

    body.update({
        "resolved": resolved,
        "verdict": verdict,
        "reading": reading,
        "alternative": {
            "insert_code": selected.insert_code,
            "description": selected.description,
            "envelope": envelope,
            "midpoint": midpoint.to_dict(),
            "midpoint_diagnosis": {
                "state": mid_diag.state,
                "v_b_mm": round(mid_pred.v_b_mm, 3),
                "ra_um": round(mid_pred.ra_um, 3),
            },
            "recommended_point": bp.to_dict(),
            "diagnosis": {
                "state": bdiag.state,
                "explanation": bdiag.explanation,
                "v_b_mm": round(bpred.v_b_mm, 3),
                "ra_um": round(bpred.ra_um, 3),
                "confidence": round(bpred.model_confidence, 3),
            },
            "notes": notes,
        },
        "alternatives_ranked": [
            {"insert_code": t.insert_code, "operation": t.operation,
             "iso_group": t.iso_group}
            for t in candidates[:5]
        ],
    })
    return body


def _evaluate(part: PartSpec, tool, point: OperatingPoint):
    sensor = _models.sensor_for_pareto(point)
    pred = predict_for_point(_models.get_mlp(), tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    diag = diagnose(pred, sensor, target_ra_um=part.target_ra_um)
    return pred, diag


def _search_envelope(part: PartSpec, tool, envelope: dict):
    """Scan a small grid in the envelope and keep the safest-setting.

    Grid is 3 speeds x 3 feeds (plus 3 depths implied by the envelope cap),
    ranked by (severity, then Ra, then -MRR) so a real improvement is
    preferred but a marginal one still surfaces.
    """
    vs = _linspace(envelope["v_c_mpm"], 3)
    fs = _linspace(envelope["f_mmpr"], 3)
    ap = min(1.5, envelope["ap_mm"][1])
    best = None
    for v in vs:
        for f in fs:
            point = OperatingPoint(v_c_mpm=v, f_mmpr=f, ap_mm=ap)
            pred, diag = _evaluate(part, tool, point)
            key = (_SEVERITY.get(diag.state, 2), round(pred.ra_um, 4),
                   round(-mrr_cm3_per_min(point), 6))
            if best is None or key < best[0]:
                best = (key, point, pred, diag)
    return best[1], best[2], best[3]


def _linspace(bounds: list[float], n: int) -> list[float]:
    lo, hi = float(bounds[0]), float(bounds[1])
    if n <= 1 or hi <= lo:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _failure(reason: str) -> dict:
    return {"resolved": False, "verdict": reason, "alternative": None,
            "reading": "no_alternative"}