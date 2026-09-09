"""Rule-based tool selection.

The prototype uses an in-process rule engine (a small, explicit
forward-chaining evaluator). The rule set lives in
``RULES`` and can be loaded from text files mirroring CLIPS syntax
later; the public API stays stable.

Each rule is a tuple ``(name, predicate, action_name)`` where
``predicate`` is a callable that returns True when the rule fires and
``action_name`` is the symbolic action to take.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from hybrid_ai.core.schemas import ISOMaterialGroup, PartSpec, ToolCandidate

#: Group-typical (V_c m/min, feed mm/rev): ranking reference per ISO
#: group, taken from the catalog mid-ranges so exotic groups are not
#: scored against steel parameters.
ISO_TARGETS: dict[ISOMaterialGroup, tuple[float, float]] = {
    "P": (220.0, 0.30),
    "M": (150.0, 0.20),
    "K": (200.0, 0.30),
    "N": (600.0, 0.20),
    "S": (60.0, 0.15),
}

Predicate = Callable[[PartSpec, ToolCandidate], bool]


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    predicate: Predicate


def _iso_match(part: PartSpec, tool: ToolCandidate) -> bool:
    return tool.iso_group == part.iso_group


def _operation_match(part: PartSpec, tool: ToolCandidate) -> bool:
    return tool.operation == part.operation


def _finish_for_steel(part: PartSpec, tool: ToolCandidate) -> bool:
    if part.iso_group != "P":
        return True
    if part.target_ra_um <= 1.0 and tool.nose_radius_mm >= 0.8:
        return True
    return part.target_ra_um > 1.0


def _heavy_roughing(part: PartSpec, tool: ToolCandidate) -> bool:
    if part.machine in {"light", "medium"}:
        return True
    if part.iso_group in {"P", "K"}:
        return tool.ap_max_mm >= 4.0
    return tool.ap_max_mm >= 2.0


def _superalloy_cautious(part: PartSpec, tool: ToolCandidate) -> bool:
    if part.iso_group != "S":
        return True
    return tool.v_c_max_mpm <= 120.0


RULES: list[Rule] = [
    Rule(
        name="iso-match",
        description="Tool ISO material group must match the part.",
        predicate=_iso_match,
    ),
    Rule(
        name="operation-match",
        description="Tool operation must match the part operation.",
        predicate=_operation_match,
    ),
    Rule(
        name="steel-finishing",
        description="For steel finishing, prefer larger nose radius.",
        predicate=_finish_for_steel,
    ),
    Rule(
        name="heavy-roughing",
        description="Heavy machines need deep-cut capable inserts.",
        predicate=_heavy_roughing,
    ),
    Rule(
        name="superalloy-cautious",
        description="Inconel/Ti: cap insert speed envelope.",
        predicate=_superalloy_cautious,
    ),
]


def build_rules() -> list[Rule]:
    """Return a fresh copy of the rule set."""
    return list(RULES)


def select_tools(
    part: PartSpec,
    catalog: Iterable[ToolCandidate],
    *,
    rules: list[Rule] | None = None,
) -> list[ToolCandidate]:
    """Apply rules and return matching tools in catalog order."""
    rules = rules or build_rules()
    candidates: list[ToolCandidate] = []
    for tool in catalog:
        if all(rule.predicate(part, tool) for rule in rules):
            candidates.append(tool)
    return candidates


def rank_tools(
    part: PartSpec,
    candidates: list[ToolCandidate],
) -> list[tuple[ToolCandidate, float]]:
    """Score candidates 0..1 by closeness of envelope to target parameters.

    The reference speed/feed are group-typical values (a steel tool at
    220 m/min scores like an aluminium tool at 600 or a superalloy
    tool at 60) instead of one global target that punishes
    non-ferrous groups. ap is scored against a 3 mm working depth.
    """

    ref_v_c, ref_f = ISO_TARGETS.get(part.iso_group, (220.0, 0.3))

    def _score(tool: ToolCandidate) -> float:
        v_c_mid = 0.5 * (tool.v_c_min_mpm + tool.v_c_max_mpm)
        f_mid = 0.5 * (tool.f_min_mmpr + tool.f_max_mmpr)
        ap_eff = min(tool.ap_max_mm, 3.0)
        v_c_term = 1.0 - min(abs(v_c_mid - ref_v_c) / ref_v_c, 1.0)
        f_term = 1.0 - min(abs(f_mid - ref_f) / ref_f, 1.0)
        ap_term = ap_eff / 3.0
        if part.target_ra_um <= 1.0:
            f_term *= 0.8
        return max(0.0, min(1.0, 0.5 * v_c_term + 0.3 * f_term + 0.2 * ap_term))

    return sorted(((t, _score(t)) for t in candidates), key=lambda x: -x[1])


def build_capp_sheet(
    part: PartSpec,
    catalog: Iterable[ToolCandidate],
) -> tuple[list[str], list[ToolCandidate], ToolCandidate | None, dict | None, list[str]]:
    """Produce a basic CAPP plan: operations, candidates, top pick, envelope, notes."""

    operations = _plan_operations(part)
    candidates = select_tools(part, catalog)
    ranked = rank_tools(part, candidates)
    selected = ranked[0][0] if ranked else None
    envelope = (
        {
            "v_c_mpm": [selected.v_c_min_mpm, selected.v_c_max_mpm],
            "f_mmpr": [selected.f_min_mmpr, selected.f_max_mmpr],
            "ap_mm": [0.5, min(selected.ap_max_mm, 3.0)],
        }
        if selected
        else None
    )
    notes = _build_notes(part, selected)
    return operations, candidates, selected, envelope, notes


def _plan_operations(part: PartSpec) -> list[str]:
    plan = ["Setup & fixturing"]
    if part.operation in {"turning", "facing"}:
        plan += ["Roughing pass", "Finishing pass", "Part off"]
    elif part.operation == "milling":
        plan += ["Adaptive roughing", "Semi-finishing", "Finishing pass", "Deburr"]
    elif part.operation == "drilling":
        plan += ["Peck drilling", "Breakthrough", "Spot face if required"]
    elif part.operation == "boring":
        plan += ["Rough boring", "Finish boring"]
    plan += ["In-process inspection", "Final QC"]
    return plan


def _build_notes(part: PartSpec, tool: ToolCandidate | None) -> list[str]:
    notes: list[str] = []
    if tool is None:
        notes.append("No catalog tool matches the part specification.")
        return notes
    if part.iso_group == "S":
        notes.append("Use rigid setup and high-pressure coolant for superalloys.")
    if part.target_ra_um <= 1.0:
        notes.append("Target finish is tight; favour larger nose radius and lower feed.")
    if part.machine == "heavy":
        notes.append("Heavy machine selected: high ap allowed, monitor spindle load.")
    notes.append(f"Recommended insert: {tool.insert_code} ({tool.description})")
    return notes
