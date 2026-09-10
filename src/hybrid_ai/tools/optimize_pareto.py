"""Tool: multi-objective cutting-parameter optimisation (NSGA-II stage)."""

from __future__ import annotations

import numpy as np

from hybrid_ai.core.schemas import PartSpec
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
                "note": "no feasible solutions"}
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
