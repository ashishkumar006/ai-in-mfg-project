from __future__ import annotations

from hybrid_ai.core.schemas import (
    FuzzyDiagnosis,
    OperatingPoint,
    ParetoSolution,
    PartSpec,
    ToolCandidate,
    WearPrediction,
)


def test_part_spec_is_frozen() -> None:
    part = PartSpec(part_id="x", material="AISI 1045", iso_group="P", operation="turning")
    assert part.iso_group == "P"
    assert part.operation == "turning"


def test_tool_candidate_round_trip() -> None:
    t = ToolCandidate(
        insert_code="X1",
        iso_group="P",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=100.0,
        v_c_max_mpm=200.0,
        f_min_mmpr=0.1,
        f_max_mmpr=0.4,
        ap_max_mm=3.0,
    )
    d = t.to_dict()
    assert d["insert_code"] == "X1"
    assert d["v_c_min_mpm"] == 100.0


def test_wear_prediction_to_dict() -> None:
    t = ToolCandidate(
        insert_code="X1", iso_group="P", operation="turning",
        nose_radius_mm=0.8, rake_angle_deg=6.0,
        v_c_min_mpm=100.0, v_c_max_mpm=200.0,
        f_min_mmpr=0.1, f_max_mmpr=0.4, ap_max_mm=3.0,
    )
    p = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.2, ap_mm=1.5)
    w = WearPrediction(tool=t, point=p, v_b_mm=0.2, ra_um=1.0)
    d = w.to_dict()
    assert d["insert_code"] == "X1"
    assert d["v_c_mpm"] == 150.0
    assert d["v_b_mm"] == 0.2


def test_pareto_solution_to_dict() -> None:
    p = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.2, ap_mm=1.5)
    s = ParetoSolution(point=p, mrr_cm3_per_min=10.0, tool_life_min=20.0, ra_um=1.0)
    d = s.to_dict()
    assert d["mrr_cm3_per_min"] == 10.0
    assert d["ra_um"] == 1.0


def test_fuzzy_diagnosis_to_dict() -> None:
    d = FuzzyDiagnosis(
        state="caution",
        wear_membership=0.3,
        roughness_membership=0.5,
        vibration_membership=0.1,
        explanation="test",
    )
    assert d.to_dict()["state"] == "caution"
