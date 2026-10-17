"""Prove each tool is wired to its stage (delegation, not duplication).

Two strategies: (1) every TOOL_SPEC required key / kwarg matches the
real run() signature; (2) tool outputs equal hand-assembled stage
calls on the same inputs.
"""

from __future__ import annotations

import inspect

import pytest

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import REGISTRY, ToolError


def _part(**kwargs) -> PartSpec:
    defaults = dict(
        part_id="X", material="m", iso_group="P", operation="turning",
        machine="medium", hardness_hb=220.0, target_ra_um=1.6, batch_size=10,
    )
    defaults.update(kwargs)
    return PartSpec(**defaults)


def test_spec_matches_run_signature() -> None:
    for name, entry in REGISTRY.items():
        params = entry["spec"]["parameters"]
        sig = inspect.signature(entry["run"])
        takes_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                           for p in sig.parameters.values())
        assert not takes_kwargs, f"{name} must have an explicit signature"
        for key in params.get("required", []):
            assert key in sig.parameters, f"{name} spec requires {key!r} run() lacks"
            assert sig.parameters[key].default is inspect.Parameter.empty, (
                f"{name}.{key} required by spec but optional in run()")
        for key, param in sig.parameters.items():
            assert key in params.get("properties", {}), (
                f"{name}.run() takes undocumented kwarg {key!r}")


def test_select_tool_matches_stage_calls(monkeypatch) -> None:
    from hybrid_ai.core.tooling_catalog import load_default_iso13399_catalog
    from hybrid_ai.stages.clips.rules import build_capp_sheet, rank_tools
    from hybrid_ai.tools import _models, select_tool

    part = _part(operation="milling", iso_group="M")
    catalog = load_default_iso13399_catalog()
    monkeypatch.setattr(_models, "get_catalog", lambda: list(catalog))
    operations, candidates, selected, envelope, notes = build_capp_sheet(part, catalog)
    body = select_tool.run(part)
    assert body["operations"] == operations
    assert body["envelope"] == envelope
    assert body["notes"] == notes
    assert [c["insert_code"] for c in body["candidates"]] == [
        t.insert_code for t, _ in rank_tools(part, candidates)]
    assert (body["selected"] or {}).get("insert_code") == (
        selected.insert_code if selected else None)


def test_predict_wear_matches_stage_call() -> None:
    from hybrid_ai.core.synthetic import build_default_tool_catalog
    from hybrid_ai.stages.ann.model import predict_for_point
    from hybrid_ai.tools import _models, predict_wear
    from hybrid_ai.tools import select_tool as _sel

    part = _part()
    point = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)
    tool = next(t for t in build_default_tool_catalog()
                if t.insert_code == _sel.run(part)["selected"]["insert_code"])
    sensor, _ = _models.resolve_sensor(point, "synthetic")
    expected = predict_for_point(_models.get_mlp(), tool, point,
                                 hardness_hb=part.hardness_hb, sensor=sensor)
    got = predict_wear.run(part, point)
    assert got["tool"] == tool.insert_code
    assert got["v_b_mm"] == round(expected.v_b_mm, 3)
    assert got["ra_um"] == round(expected.ra_um, 3)


def test_diagnose_state_matches_stage_calls() -> None:
    from hybrid_ai.core.synthetic import build_default_tool_catalog
    from hybrid_ai.stages.ann.model import predict_for_point
    from hybrid_ai.stages.fuzzy.diagnoser import diagnose
    from hybrid_ai.tools import _models, diagnose_state

    part = _part()
    point = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)
    tool = build_default_tool_catalog()[0]
    sensor, _ = _models.resolve_sensor(point, "synthetic")
    pred = predict_for_point(_models.get_mlp(), tool, point,
                             hardness_hb=part.hardness_hb, sensor=sensor)
    expected = diagnose(pred, sensor, target_ra_um=part.target_ra_um)
    got = diagnose_state.run(part, point, insert_code=tool.insert_code)
    assert got["state"] == expected.state
    assert got["explanation"] == expected.explanation
    assert got["v_b_mm"] == round(pred.v_b_mm, 3)


def test_optimize_matches_direct_call() -> None:
    from hybrid_ai.stages.nsga2.optimiser import NsgaConfig, optimise
    from hybrid_ai.tools import _models, optimize_pareto, select_tool

    part = _part()
    code = select_tool.run(part)["selected"]["insert_code"]
    tool = next(t for t in _models.get_catalog() if t.insert_code == code)
    expected = optimise(
        _models.get_mlp(), tool, hardness_hb=part.hardness_hb,
        config=NsgaConfig(pop_size=10, n_generations=4, seed=7,
                          enforce_safe=True),
        sensor_factory=_models.sensor_for_pareto,
        target_ra_um=part.target_ra_um)
    got = optimize_pareto.run(part, insert_code=code, pop_size=10,
                              n_generations=4, seed=7, safe_only=True)
    assert [s["v_c_mpm"] for s in got["solutions"]] == [
        s.point.v_c_mpm for s in expected]


def test_compare_mlp_row_matches_predict_tool() -> None:
    from hybrid_ai.tools import compare_models, predict_wear

    part = _part()
    point = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)
    solo = predict_wear.run(part, point)
    rows = {r["model"]: r for r in compare_models.run(part, point)["rows"]}
    assert rows["MLP (blended)"]["v_b_mm"] == solo["v_b_mm"]
    assert rows["MLP (blended)"]["ra_um"] == solo["ra_um"]


def test_resolve_tool_error_codes(monkeypatch) -> None:
    from hybrid_ai.tools import _models

    with pytest.raises(ToolError) as exc_info:
        _models.resolve_tool(_part(), "NOPE")
    assert exc_info.value.code == "unknown_tool"
    mini = [t for t in _models.get_catalog()
            if t.iso_group == "P" and t.operation == "turning"]
    monkeypatch.setattr(_models, "get_catalog", lambda: list(mini))
    with pytest.raises(ToolError) as exc_info:
        _models.resolve_tool(_part(iso_group="S", operation="drilling"), None)
    assert exc_info.value.code == "no_match"
    with pytest.raises(ToolError) as exc_info:
        _models.resolve_sensor(OperatingPoint(200, 0.2, 1.0), "radar")
    assert exc_info.value.code == "bad_sensor_source"


def test_nested_enum_violation_names_field() -> None:
    from hybrid_ai.tools import validate_args

    with pytest.raises(ToolError) as exc_info:
        validate_args(REGISTRY["select_tool"]["spec"],
                      {"part": {"part_id": "X", "material": "m",
                                "iso_group": "Z", "operation": "turning"}})
    assert "iso_group" in str(exc_info.value)


def test_list_tools_matches_repo_filter() -> None:
    from hybrid_ai.core.sqlite_loader import ToolCatalogRepository
    from hybrid_ai.core.synthetic import build_default_tool_catalog
    from hybrid_ai.tools import catalog

    repo = ToolCatalogRepository(":memory:")
    repo.upsert(build_default_tool_catalog())
    expected = repo.filter(iso_group="P", operation="turning")
    got = catalog.run_list(iso_group="P", operation="turning")["tools"]
    assert [t["insert_code"] for t in got] == [t.insert_code for t in expected]
