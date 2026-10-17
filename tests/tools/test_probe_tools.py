from __future__ import annotations

import pytest

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds
from hybrid_ai.tools import REGISTRY, ToolError, validate_args


def _part(**kwargs) -> PartSpec:
    defaults = dict(
        part_id="X", material="m", iso_group="P", operation="turning",
        machine="medium", hardness_hb=220.0, target_ra_um=1.6, batch_size=10,
    )
    defaults.update(kwargs)
    return PartSpec(**defaults)


def _point() -> OperatingPoint:
    return OperatingPoint(v_c_mpm=250.0, f_mmpr=0.30, ap_mm=1.75)


def test_sweep_is_deterministic_and_read_only() -> None:
    from hybrid_ai.tools import sweep_pareto

    part = _part()
    first = sweep_pareto.run(part, n_seeds=2, seed_base=7,
                             pop_size=8, n_generations=3)
    second = sweep_pareto.run(part, n_seeds=2, seed_base=7,
                              pop_size=8, n_generations=3)
    assert first["seeds"] == [7, 8]
    assert len(first["per_seed"]) == 2
    assert first["per_seed"] == second["per_seed"]
    assert first["verdict"] in ("stable", "check", "unstable")
    assert first["mutated"] is False


def test_sweep_rejects_bad_budget() -> None:
    spec = REGISTRY["sweep_pareto"]["spec"]
    with pytest.raises(ToolError):
        validate_args(spec, {"part": {}, "n_seeds": 9})


def test_sensitivity_probes_all_axes() -> None:
    from hybrid_ai.tools import sensitivity_probe

    out = sensitivity_probe.run(_part(), _point(), rel_delta=0.1)
    assert len(out["probes"]) == 6
    assert {p["axis"] for p in out["probes"]} == {"v_c_mpm", "f_mmpr",
                                                 "ap_mm"}
    assert out["most_sensitive_axis"] in ("v_c_mpm", "f_mmpr", "ap_mm")
    assert out["mutated"] is False
    plus = next(p for p in out["probes"]
                if p["axis"] == "f_mmpr" and p["direction"] == "+")
    assert plus["point"]["f_mmpr"] == pytest.approx(0.33)


def test_propose_never_applies_and_keeps_production_thresholds() -> None:
    from hybrid_ai.tools import propose_recalibration

    before = FuzzyThresholds.for_target(1.6)
    out = propose_recalibration.run(target_ra_um=1.6, v_b_low=0.10)
    assert out["mode"] == "proposal-only"
    assert out["applied"] is False
    assert out["mutated"] is False
    assert out["changed_axes"] == ["v_b_low"]
    assert out["n_flips"] == sum(out["flip_counts"].values())
    assert "pytest" in out["apply_instructions"]
    assert FuzzyThresholds.for_target(1.6) == before


def test_propose_identity_candidate_flips_nothing() -> None:
    from hybrid_ai.tools import propose_recalibration

    out = propose_recalibration.run(target_ra_um=1.6)
    assert out["n_flips"] == 0
    assert out["changed_axes"] == []


def test_propose_rejects_broken_ordering_and_unknown_axes() -> None:
    from hybrid_ai.tools import propose_recalibration

    with pytest.raises(ToolError):
        propose_recalibration.run(target_ra_um=1.6, v_b_low=0.99,
                                  v_b_mid=0.30)
    with pytest.raises(TypeError):
        propose_recalibration.run(target_ra_um=1.6, nonsense_axis=1.0)  # type: ignore[call-arg]
    with pytest.raises(ToolError):
        validate_args(REGISTRY["propose_recalibration"]["spec"],
                      {"bogus_param": 1})
