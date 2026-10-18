from __future__ import annotations

import pytest

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import REGISTRY, ToolError, validate_args


def _part(**kwargs) -> PartSpec:
    defaults = dict(
        part_id="X", material="m", iso_group="P", operation="turning",
        machine="medium", hardness_hb=220.0, target_ra_um=1.6, batch_size=10,
    )
    defaults.update(kwargs)
    return PartSpec(**defaults)


def test_registry_has_expected_specs() -> None:
    assert set(REGISTRY) == {
        "select_tool", "predict_wear", "compare_models", "diagnose_state",
        "optimize_pareto", "list_tools", "model_benchmarks",
        "sweep_pareto", "sensitivity_probe", "propose_recalibration",
        "replan",
    }
    for name, entry in REGISTRY.items():
        spec = entry["spec"]
        assert spec["name"] == name
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"
        assert callable(entry["run"])


def test_validate_args_accepts_and_rejects() -> None:
    spec = REGISTRY["select_tool"]["spec"]
    good = {"part": {"part_id": "X", "material": "m", "iso_group": "P",
                     "operation": "drilling"}}
    assert validate_args(spec, good) == good
    with pytest.raises(ToolError):
        validate_args(spec, {"material": "m"})
    with pytest.raises(ToolError):
        validate_args(spec, {"part": "not-an-object"})
    with pytest.raises(ToolError) as exc_info:
        validate_args(spec, {"part": {"part_id": "X"}})
    assert "part.part_id" not in str(exc_info.value)
    assert "material" in str(exc_info.value)
    with pytest.raises(ToolError):
        validate_args(REGISTRY["optimize_pareto"]["spec"],
                      {"part": {}, "pop_size": 5})
    with pytest.raises(ToolError) as exc_info:
        validate_args(spec, {**good, "bogus_param": 1})
    assert "unknown argument" in str(exc_info.value)


def test_select_tool_match_and_no_match() -> None:
    from hybrid_ai.tools import select_tool

    hit = select_tool.run(_part(operation="drilling"))
    assert hit["selected"]["insert_code"] == "SDMX090308-P2"
    assert hit["envelope"] is not None
    setup = hit["setup"]
    assert setup["operation"] == "drilling"
    assert "bushing" in setup["workholding"].lower()
    assert setup["steps"] and setup["datums"] and setup["clamping"]


def test_select_tool_no_match_on_pared_catalog(monkeypatch) -> None:
    """The no-match diagnostics stay reachable even though the bundled
    catalog now satisfies every (group, operation, severity) cell."""
    from hybrid_ai.tools import _models, select_tool

    mini = [t for t in _models.get_catalog()
            if t.iso_group == "P" and t.operation == "turning"]
    monkeypatch.setattr(_models, "get_catalog", lambda: list(mini))
    miss = select_tool.run(_part(iso_group="S", operation="drilling"))
    assert miss["selected"] is None
    assert "setup" not in miss
    assert miss["no_match"]["per_rule"] and miss["no_match"]["closest"]


def test_predict_and_diagnose_state() -> None:
    from hybrid_ai.tools import diagnose_state, predict_wear

    part = _part()
    point = OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)
    pred = predict_wear.run(part, point)
    assert pred["v_b_mm"] > 0 and pred["ra_um"] > 0
    diag = diagnose_state.run(part, point)
    assert diag["state"] in {"safe", "caution", "critical"}
    assert diag["explanation"] and diag["sensor_source"] == "synthetic"
    diag_m = diagnose_state.run(part, point, sensor_source="measured")
    assert diag_m["sensor_source"] == "measured NASA run"


def test_optimize_pareto_safe_only() -> None:
    from hybrid_ai.tools import diagnose_state, optimize_pareto

    front = optimize_pareto.run(_part(), pop_size=12, n_generations=5,
                                safe_only=True)
    assert front["safe_only"] is True
    assert len(front["solutions"]) >= 1
    assert set(front["extremes"]) == {"max_mrr", "max_life", "min_ra"}


def test_no_safe_front_explains_itself_honestly() -> None:
    """A part whose whole envelope is unsafe must report
    ``no_safe_setting_exists``, not a search miss, and the empty path
    must still carry the metric keys.

    The case: a 0.3 um finish asked of an insert whose minimum feed
    (0.15 mm/rev, 0.8 mm nose) already leaves ~0.7 um of kinematic
    cusp height — no feed in the envelope can pass the Ra gate.
    """
    from hybrid_ai.tools import optimize_pareto

    part = _part(target_ra_um=0.3)
    out = optimize_pareto.run(part, pop_size=8, n_generations=3)
    assert out["solutions"] == []
    nsf = out["no_safe_front"]
    assert nsf["reason"] == "no_safe_setting_exists"
    assert out["spread"] is None
    assert out["hypervolume"] is None


def test_corner_probe_reports_search_miss_not_no_safe_point() -> None:
    """Regression: the no-front explainer's first candidate (the envelope
    corner) used to be excluded from its own safe check, so a genuinely
    safe corner was reported as ``no_safe_setting_exists``."""
    from hybrid_ai.stages.clips.rules import build_capp_sheet
    from hybrid_ai.tools import _models
    from hybrid_ai.tools.optimize_pareto import _explain_no_front

    part = _part()
    tool = _models.resolve_tool(part, None)
    out = _explain_no_front(part, tool, safe_only=True)
    assert out["reason"] == "search_missed_safe_point"
    assert out["diagnosis"]["state"] == "safe"
    _, _, _, env, _ = build_capp_sheet(part, _models.get_catalog())
    assert out["point"] == {"v_c_mpm": env["v_c_mpm"][0],
                            "f_mmpr": env["f_mmpr"][0],
                            "ap_mm": env["ap_mm"][0]}


def test_list_tools_filter() -> None:
    from hybrid_ai.core.tooling_catalog import load_default_iso13399_catalog
    from hybrid_ai.tools import catalog

    all_tools = catalog.run_list()
    assert len(all_tools["tools"]) == len(load_default_iso13399_catalog())
    drills = catalog.run_list(operation="drilling")
    assert drills["tools"] and all(t["operation"] == "drilling"
                                   for t in drills["tools"])


def test_model_benchmarks_structure() -> None:
    from hybrid_ai.tools import catalog

    rows = catalog.run_benchmarks()["rows"]
    assert {r["model"] for r in rows} >= {"MLP", "CNN-BiLSTM", "HGBR"}
    for r in rows:
        if "error" in r:
            continue
        assert isinstance(r.get("holdout"), bool), f"{r['model']} has no holdout flag"
        assert r.get("eval"), f"{r['model']} has no eval label"
    assert any(r.get("holdout") is True for r in rows), "needs a genuine holdout row"
    assert any(r.get("holdout") is False for r in rows), "train-set rows must be labelled"


def test_capp_matrix_selected_xor_explained() -> None:
    from hybrid_ai.core.schemas import PartSpec
    from hybrid_ai.tools import select_tool

    matched, empty = 0, 0
    for op in ("turning", "facing", "milling", "drilling", "boring"):
        for iso in ("P", "M", "K", "N", "S"):
            for machine in ("light", "medium", "heavy"):
                part = PartSpec(part_id="T", material="m", iso_group=iso,
                                operation=op, machine=machine, hardness_hb=220.0,
                                target_ra_um=1.6, batch_size=10)
                body = select_tool.run(part)
                if body["selected"] is None:
                    empty += 1
                    assert body["envelope"] is None
                    assert body["no_match"]["per_rule"]
                    assert body["no_match"]["closest"]
                else:
                    matched += 1
                    assert body["envelope"] is not None
                    assert body["selected"]["operation"] == op
                    assert body["selected"]["iso_group"] == iso
    assert matched == 75 and empty == 0
