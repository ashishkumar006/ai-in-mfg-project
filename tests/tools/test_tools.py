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
    assert hit["selected"]["insert_code"] == "SDMX090308-P"
    assert hit["envelope"] is not None
    miss = select_tool.run(_part(iso_group="S", operation="drilling"))
    assert miss["selected"] is None
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


def test_list_tools_filter() -> None:
    from hybrid_ai.tools import catalog

    all_tools = catalog.run_list()
    assert len(all_tools["tools"]) == 15
    drills = catalog.run_list(operation="drilling")
    assert drills["tools"] and all(t["operation"] == "drilling"
                                   for t in drills["tools"])


def test_model_benchmarks_structure() -> None:
    from hybrid_ai.tools import catalog

    rows = catalog.run_benchmarks()["rows"]
    assert {r["model"] for r in rows} >= {"MLP", "CNN-BiLSTM", "HGBR"}


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
    assert matched > 0 and empty > 0
