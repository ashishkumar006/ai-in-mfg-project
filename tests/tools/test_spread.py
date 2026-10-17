from __future__ import annotations

from hybrid_ai.core.schemas import PartSpec
from hybrid_ai.tools import _models
from hybrid_ai.tools.optimize_pareto import run as optimize


def _part(**kw) -> PartSpec:
    d = dict(part_id="S", material="AISI 4140", iso_group="P", operation="turning",
             machine="medium", hardness_hb=220.0, target_ra_um=1.6, batch_size=5)
    d.update(kw)
    return PartSpec(**d)


def test_spread_reports_spacing_and_extent() -> None:
    part = _part()
    tool = _models.resolve_tool(part, None)
    out = optimize(part, tool.insert_code, pop_size=16, n_generations=8, seed=7)
    sp = out["spread"]
    assert sp is not None
    assert sp["n_points"] == len(out["solutions"])
    assert sp["spacing"] >= 0.0
    assert len(sp["extent"]) == 3 and all(v >= 0.0 for v in sp["extent"])


def test_spread_is_none_for_empty_front() -> None:
    part = _part(iso_group="S", machine="heavy", hardness_hb=410.0,
                 target_ra_um=0.4)
    tool = _models.resolve_tool(part, None)
    out = optimize(part, tool.insert_code, pop_size=10, n_generations=4, seed=7)
    if not out["solutions"]:
        assert out["spread"] is None
