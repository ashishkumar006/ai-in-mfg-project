"""Tool: seed-sweep probe over the Pareto optimiser (read-only).

Re-runs :mod:`hybrid_ai.tools.optimize_pareto` with successive seeds and
compares the extremes, so the agent can tell a stable trade-off front
from a seed-lucky one before reporting. Never writes state: every run
goes through the normal read-only optimise path.
"""

from __future__ import annotations

from hybrid_ai.core.schemas import PartSpec
from hybrid_ai.tools import optimize_pareto
from hybrid_ai.tools.schemas import PART_SCHEMA

TOOL_SPEC: dict = {
    "name": "sweep_pareto",
    "description": ("Probe optimiser stability: re-run the safe Pareto "
                    "search over successive seeds and compare the extremes "
                    "(max MRR, max life, min Ra). Read-only; reports spread "
                    "so the agent can distrust seed-lucky fronts."),
    "parameters": {
        "type": "object",
        "properties": {
            "part": PART_SCHEMA,
            "insert_code": {"type": ["string", "null"]},
            "n_seeds": {"type": "integer", "minimum": 1, "maximum": 5,
                        "default": 3},
            "seed_base": {"type": "integer", "default": 42},
            "pop_size": {"type": "integer", "minimum": 8, "maximum": 200,
                         "default": 60},
            "n_generations": {"type": "integer", "minimum": 3, "maximum": 200,
                              "default": 40},
            "safe_only": {"type": "boolean", "default": True},
        },
        "required": ["part"],
    },
}


def _spread(values: list[float]) -> float:
    """Range of a value list, 0.0 when empty."""
    return float(max(values) - min(values)) if values else 0.0


def run(part: PartSpec, insert_code: str | None = None,
        n_seeds: int = 3, seed_base: int = 42,
        pop_size: int = 60, n_generations: int = 40,
        safe_only: bool = True) -> dict:
    """Sweep seeds and return per-seed extremes plus spread summary."""
    seeds = [int(seed_base) + i for i in range(int(n_seeds))]
    per_seed = []
    for seed in seeds:
        front = optimize_pareto.run(
            part, insert_code=insert_code, pop_size=int(pop_size),
            n_generations=int(n_generations), seed=seed, safe_only=safe_only,
        )
        extremes = front.get("extremes", {})
        per_seed.append({
            "seed": seed,
            "tool": front.get("tool"),
            "n_solutions": len(front.get("solutions", [])),
            "max_mrr": (extremes.get("max_mrr") or {}).get("mrr_cm3_per_min"),
            "max_life": (extremes.get("max_life") or {}).get("tool_life_min"),
            "min_ra": (extremes.get("min_ra") or {}).get("ra_um"),
        })
    mrrs = [r["max_mrr"] for r in per_seed if r["max_mrr"] is not None]
    lifes = [r["max_life"] for r in per_seed if r["max_life"] is not None]
    ras = [r["min_ra"] for r in per_seed if r["min_ra"] is not None]
    mean_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    rel = (_spread(mrrs) / mean_mrr) if mean_mrr > 0 else 0.0
    return {
        "tool": per_seed[0]["tool"] if per_seed else insert_code,
        "safe_only": safe_only,
        "seeds": seeds,
        "per_seed": per_seed,
        "spread": {
            "max_mrr_range": round(_spread(mrrs), 3),
            "max_life_range": round(_spread(lifes), 3),
            "min_ra_range": round(_spread(ras), 3),
            "relative_mrr_spread": round(rel, 3),
        },
        "verdict": ("stable" if rel < 0.05 else
                    "check" if rel < 0.20 else "unstable"),
        "mutated": False,
    }
