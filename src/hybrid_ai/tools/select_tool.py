"""Tool: rule-based insert selection + CAPP plan (CLIPS stage)."""

from __future__ import annotations

from hybrid_ai.core.schemas import PartSpec
from hybrid_ai.stages.clips.rules import (
    build_capp_sheet,
    build_rules,
    build_setup_plan,
    rank_tools,
)
from hybrid_ai.tools import _models
from hybrid_ai.tools.schemas import PART_SCHEMA

TOOL_SPEC: dict = {
    "name": "select_tool",
    "description": ("Pick the best catalog insert for a part spec and return "
                    "the CAPP operation plan, a setup & fixturing plan "
                    "(workholding, datums, clamping, tool holding, steps), "
                    "ranked candidates, selected tool, "
                    "recommended envelope and planner notes. When nothing "
                    "matches, explains per-rule survival counts plus closest inserts."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
        },
        "required": ["part"],
    },
}


def run(part: PartSpec) -> dict:
    """Run tool selection for a part spec."""
    catalog = _models.get_catalog()
    operations, candidates, selected, envelope, notes = build_capp_sheet(part, catalog)
    ranked = rank_tools(part, candidates)
    body: dict = {
        "operations": operations,
        "candidates": [{**t.to_dict(), "score": round(s, 3)} for t, s in ranked],
        "selected": selected.to_dict() if selected else None,
        "envelope": envelope,
        "notes": notes,
    }
    if selected is not None:
        body["setup"] = build_setup_plan(part, selected)
    if selected is None:
        rules = build_rules()
        body["no_match"] = {
            "per_rule": [
                {"rule": r.name,
                 "passing": sum(1 for t in catalog if r.predicate(part, t)),
                 "requires": r.description}
                for r in rules
            ],
            "closest": [
                {"insert_code": t.insert_code, "iso_group": t.iso_group,
                 "operation": t.operation, "score": round(s, 3)}
                for t, s in rank_tools(part, list(catalog))[:3]
            ],
        }
    return body
