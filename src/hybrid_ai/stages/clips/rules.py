"""Rule-based tool selection, executed by a real CLIPS engine (clipspy).

Stage 1 of the pipeline is CLIPS. Selection is a genuine forward-chaining
inference: part and tool facts are asserted, a set of *reject* rules
asserts ``(rejected ?code)`` for ineligible inserts, and a final
``eligible`` rule asserts ``(candidate ?code)`` for everything not
rejected. This is the CLIPS engine doing the work, not Python pretending
to be one.

The ``Rule`` objects in :data:`RULES` are the human-readable specification
of the same five constraints; their predicates are used for the per-rule
diagnostics shown when nothing matches, and a cross-check test asserts the
CLIPS result and the predicate result agree for a matrix of parts.
"""

from __future__ import annotations

import clips
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

#: CLIPS source for the selection rules. Each of the five selection
#: constraints appears as a *reject* rule, so a tool is kept exactly when
#: every constraint holds — the same semantics as the predicate list.
CLIPS_RULES = r"""
(deftemplate part
  (slot iso) (slot operation) (slot target_ra) (slot machine))
(deftemplate tool
  (slot code) (slot iso) (slot operation)
  (slot nose) (slot vc_max) (slot ap_max))
(deftemplate rejected (slot code))
(deftemplate candidate (slot code))

(defrule reject-iso
  (part (iso ?p))
  (tool (code ?c) (iso ?ti))
  (test (neq ?ti ?p))
  => (assert (rejected (code ?c))))

(defrule reject-operation
  (part (operation ?o))
  (tool (code ?c) (operation ?to))
  (test (neq ?to ?o))
  => (assert (rejected (code ?c))))

(defrule reject-steel-finish
  (part (iso "P") (target_ra ?ra))
  (tool (code ?c) (nose ?n))
  (test (<= ?ra 1.0))
  (test (< ?n 0.8))
  => (assert (rejected (code ?c))))

(defrule reject-heavy-ferrous
  (part (machine "heavy") (iso ?iso))
  (tool (code ?c) (ap_max ?ap))
  (test (or (eq ?iso "P") (eq ?iso "K")))
  (test (< ?ap 4.0))
  => (assert (rejected (code ?c))))

(defrule reject-heavy-other
  (part (machine "heavy") (iso ?iso))
  (tool (code ?c) (ap_max ?ap))
  (test (and (neq ?iso "P") (neq ?iso "K")))
  (test (< ?ap 2.0))
  => (assert (rejected (code ?c))))

(defrule reject-superalloy
  (part (iso "S"))
  (tool (code ?c) (vc_max ?v))
  (test (> ?v 120.0))
  => (assert (rejected (code ?c))))

(defrule eligible
  (tool (code ?c))
  (not (rejected (code ?c)))
  => (assert (candidate (code ?c))))
"""


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


def _assert_facts(env: clips.Environment, part: PartSpec,
                  catalog: Iterable[ToolCandidate]) -> None:
    env.assert_string(
        f'(part (iso "{part.iso_group}") (operation "{part.operation}") '
        f"(target_ra {float(part.target_ra_um)!r}) "
        f'(machine "{part.machine}"))'
    )
    for tool in catalog:
        env.assert_string(
            f'(tool (code "{tool.insert_code}") (iso "{tool.iso_group}") '
            f'(operation "{tool.operation}") '
            f"(nose {float(tool.nose_radius_mm)!r}) "
            f"(vc_max {float(tool.v_c_max_mpm)!r}) "
            f"(ap_max {float(tool.ap_max_mm)!r}))"
        )


def _load_rules(env: clips.Environment) -> None:
    """Load the rule base into an environment.

    ``Environment.build()`` compiles exactly one construct, so the rule
    source is split into balanced top-level blocks first. A malformed
    block fails loudly here rather than silently in production.
    """
    blocks: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(CLIPS_RULES):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                blocks.append(CLIPS_RULES[start:i + 1])
            elif depth < 0:
                raise ValueError("CLIPS rule source has unbalanced parentheses")
    for block in blocks:
        if block.strip():
            env.build(block)


def clips_select_codes(part: PartSpec,
                       catalog: Iterable[ToolCandidate]) -> list[str]:
    """Run the CLIPS rule base and return the surviving insert codes.

    Codes come back in catalog order, so callers keep a stable,
    deterministic ordering.
    """
    catalog = list(catalog)
    env = clips.Environment()
    _load_rules(env)
    _assert_facts(env, part, catalog)
    env.run()
    codes: list[str] = []
    for fact in env.facts():
        if fact.template.name == "candidate":
            codes.append(str(fact["code"]))
    env.clear()
    order = {t.insert_code: i for i, t in enumerate(catalog)}
    return sorted(codes, key=lambda c: order.get(c, len(order)))


def _predicate_select(part: PartSpec,
                      catalog: list[ToolCandidate],
                      rules: list[Rule]) -> list[ToolCandidate]:
    return [t for t in catalog
            if all(rule.predicate(part, t) for rule in rules)]


def select_tools(
    part: PartSpec,
    catalog: Iterable[ToolCandidate],
    *,
    rules: list[Rule] | None = None,
) -> list[ToolCandidate]:
    """Apply the rules and return matching tools in catalog order.

    With the default rule set the selection is performed by CLIPS. If an
    explicit ``rules`` list is passed, the predicate-based filter is used
    instead (the per-rule diagnostics path and the tests use this).
    """
    catalog = list(catalog)
    if rules is not None:
        return _predicate_select(part, catalog, rules)
    survivors = set(clips_select_codes(part, catalog))
    return [t for t in catalog if t.insert_code in survivors]


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


def build_setup_plan(part: PartSpec, tool: ToolCandidate | None) -> dict:
    """Setup & fixturing plan for a part (deliverable 1: process planning).

    Rules off the operation, machine class, material group and finish
    target: a first-cut plan for a planner to review, not a fixture design.

    Parameters
    ----------
    part:
        Part specification driving the plan.
    tool:
        Selected insert, or ``None`` when the catalog has no match.

    Returns
    -------
    dict
        Workholding, datums, clamping, tool holding, an ordered ``steps``
        list, ``setup_count`` and planner ``notes``.
    """
    holder = tool.insert_code if tool is not None else "the selected insert"
    op = part.operation

    if op in {"turning", "facing"}:
        plan: dict = {
            "workholding": (
                "3-jaw self-centring chuck for chucked work, or a collet chuck "
                "for bar stock; soft jaws once the diameter is finished"
            ),
            "datums": (
                "Z0 on the finished face, X0 on the spindle axis; face and "
                "centre the blank before the first pass"
            ),
            "clamping": (
                "clamp on the largest rigid diameter; keep overhang under "
                "3x diameter"
            ),
            "tool_holding": f"ISO 5608 turning holder for {holder}, set on centre",
            "setup_count": 2 if op == "facing" else 1,
        }
        steps = [
            "Load and indicate the blank (runout < 0.02 mm)",
            "Face and centre, set Z0",
            "Rough, leaving 0.3 mm finish allowance",
            "Finish with the selected insert",
            "Part off / second operation if required",
        ]
    elif op == "drilling":
        plan = {
            "workholding": (
                "machine vise or fixture plate with a sacrificial backing "
                "plate; drill bushings for repeat batches"
            ),
            "datums": "X/Y from two edges or a pinned hole, Z0 on the top face",
            "clamping": (
                "back the exit face to limit breakout burrs; keep clamps clear "
                "of the hole pattern"
            ),
            "tool_holding": f"collet or drill chuck for {holder}, shortest gauge length",
            "setup_count": 1,
        }
        steps = [
            "Load the blank on the backing plate",
            "Probe X/Y and the top face",
            "Peck drill per the cutting plan",
            "Break through at reduced feed",
            "Spot-face / deburr if required",
        ]
    elif op == "boring":
        plan = {
            "workholding": (
                "chuck or fixture plate with bored soft jaws holding on the "
                "finished diameter"
            ),
            "datums": "Z0 on the finished face, bore axis as X0/Y0",
            "clamping": "clamp on the outer diameter only; do not distort the bore",
            "tool_holding": f"boring head or boring bar holder for {holder}",
            "setup_count": 1,
        }
        steps = [
            "Load and indicate the part (runout < 0.02 mm)",
            "Set the bore axis from the datum",
            "Rough bore, leaving 0.2 mm",
            "Finish bore and measure",
            "Deburr and inspect",
        ]
    else:
        plan = {
            "workholding": (
                "precision machine vise on a fixture plate; low-profile clamps "
                "on a sub-plate for thin or long parts"
            ),
            "datums": (
                "Z0 on the top face, X/Y from two machined edges; probe the "
                "blank before roughing"
            ),
            "clamping": (
                "support directly under the cut; keep clamps clear of faces "
                "machined in the same setup"
            ),
            "tool_holding": f"BT40/CAT40 collet or side-lock holder for {holder}",
            "setup_count": 1,
        }
        steps = [
            "Mount and indicate the vise or sub-plate square",
            "Load the blank with 3-5 mm above the jaws, probe X/Y/Z",
            "Adaptive roughing with coolant, leaving 0.3 mm",
            "Semi-finish and finish walls and floor",
            "Deburr and inspect before unloading",
        ]

    notes: list[str] = []
    if part.machine == "light":
        notes.append("Light machine: keep ap inside the envelope and add a sub-plate for rigidity.")
    elif part.machine == "heavy":
        notes.append("Heavy machine: preload the clamps and watch spindle load near the envelope top.")
    if part.iso_group == "S":
        notes.append("Superalloy: rigid setup plus high-pressure coolant through the tool if available.")
    if part.target_ra_um <= 1.0:
        notes.append("Tight finish: clean jaws/vise faces and take a light spring pass before the finish cut.")

    plan.update({"operation": op, "machine_class": part.machine,
                 "steps": steps, "notes": notes})
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
