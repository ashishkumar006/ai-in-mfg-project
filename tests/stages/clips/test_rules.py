from __future__ import annotations

from hybrid_ai.stages.clips.rules import build_capp_sheet, build_rules, rank_tools, select_tools
from hybrid_ai.core.schemas import PartSpec
from hybrid_ai.core.synthetic import build_default_tool_catalog


def _spec(**kwargs) -> PartSpec:
    defaults = dict(
        part_id="X",
        material="AISI 1045",
        iso_group="P",
        operation="turning",
        machine="medium",
        hardness_hb=200.0,
        target_ra_um=1.6,
        batch_size=1,
    )
    defaults.update(kwargs)
    return PartSpec(**defaults)


def test_rules_present() -> None:
    rules = build_rules()
    names = {r.name for r in rules}
    assert "iso-match" in names
    assert "superalloy-cautious" in names


def test_iso_match_filters_correctly() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="S")
    matches = select_tools(p, catalog)
    assert all(t.iso_group == "S" for t in matches)
    assert len(matches) >= 1


def test_superalloy_rule() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="S", operation="turning")
    matches = select_tools(p, catalog)
    for t in matches:
        assert t.v_c_max_mpm <= 120.0


def test_rank_tools_returns_scores_in_unit_interval() -> None:
    catalog = build_default_tool_catalog()
    p = _spec()
    candidates = select_tools(p, catalog)
    ranked = rank_tools(p, candidates)
    assert ranked
    for _, score in ranked:
        assert 0.0 <= score <= 1.0


def test_capp_sheet_for_steel() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="P", operation="turning")
    ops, cands, selected, envelope, notes = build_capp_sheet(p, catalog)
    assert "Roughing pass" in " ".join(ops)
    assert selected is not None
    assert envelope is not None
    assert any("insert" in n.lower() for n in notes)


def test_steel_turning_still_picks_pm_insert() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="P", operation="turning")
    _, _, selected, _, _ = build_capp_sheet(p, catalog)
    assert selected is not None
    assert selected.insert_code == "CNMG120408-PM"


def test_iso_aware_ranking_prefers_group_typical_speed() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="N", operation="milling")
    ranked = rank_tools(p, select_tools(p, catalog))
    assert ranked and ranked[0][0].insert_code == "APMT1604PDER-N1"


def test_every_operation_matches_steel_on_medium() -> None:
    catalog = build_default_tool_catalog()
    for operation in ("turning", "facing", "milling", "drilling", "boring"):
        p = _spec(iso_group="P", operation=operation)
        matches = select_tools(p, catalog)
        assert matches, f"no tool for P/{operation}"
        assert all(t.operation == operation for t in matches)


def test_drilling_selects_drill() -> None:
    catalog = build_default_tool_catalog()
    p = _spec(iso_group="P", operation="drilling")
    ops, cands, selected, envelope, notes = build_capp_sheet(p, catalog)
    assert selected is not None
    assert selected.insert_code == "SDMX090308-P"
    assert "drilling" in " ".join(ops).lower()


def test_heavy_machine_matches_non_steel() -> None:
    catalog = build_default_tool_catalog()
    for iso_group, operation in (("M", "drilling"), ("S", "turning")):
        p = _spec(iso_group=iso_group, operation=operation, machine="heavy")
        matches = select_tools(p, catalog)
        assert matches, f"no tool for heavy {iso_group}/{operation}"
