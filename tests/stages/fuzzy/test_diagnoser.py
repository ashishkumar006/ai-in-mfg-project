from __future__ import annotations

from pytest import approx

from hybrid_ai.core.schemas import OperatingPoint, ToolCandidate, WearPrediction
from hybrid_ai.core.synthetic import generate_synthetic_sensor
from hybrid_ai.stages.fuzzy.diagnoser import FuzzyThresholds, diagnose, state_is_safe


def _pred(v_b: float, ra: float) -> WearPrediction:
    t = ToolCandidate(
        insert_code="T", iso_group="P", operation="turning",
        nose_radius_mm=0.8, rake_angle_deg=6.0,
        v_c_min_mpm=100.0, v_c_max_mpm=200.0,
        f_min_mmpr=0.1, f_max_mmpr=0.4, ap_max_mm=3.0,
    )
    p = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.2, ap_mm=1.5)
    return WearPrediction(tool=t, point=p, v_b_mm=v_b, ra_um=ra)


def test_safe_when_low_signals() -> None:
    diag = diagnose(_pred(v_b=0.05, ra=0.4), thresholds=FuzzyThresholds())
    assert diag.state == "safe"
    assert state_is_safe(diag)


def test_critical_when_high_wear() -> None:
    diag = diagnose(_pred(v_b=0.6, ra=0.4), thresholds=FuzzyThresholds())
    assert diag.state == "critical"


def test_critical_when_high_roughness() -> None:
    diag = diagnose(_pred(v_b=0.1, ra=3.5), thresholds=FuzzyThresholds())
    assert diag.state == "critical"


def test_vibration_can_trigger_critical() -> None:
    sensor = generate_synthetic_sensor(v_c=300.0, f=0.4, ap=3.0, seed=11)
    diag = diagnose(_pred(v_b=0.1, ra=0.4), sensor=sensor, thresholds=FuzzyThresholds())
    assert diag.state in {"caution", "critical"}


def test_explanation_is_non_empty() -> None:
    diag = diagnose(_pred(v_b=0.5, ra=0.4), thresholds=FuzzyThresholds())
    assert diag.explanation


def test_membership_spot_values() -> None:
    import numpy as np

    from hybrid_ai.stages.fuzzy.diagnoser import _trap_hi, _tri

    assert float(_tri(np.array([0.0]), 0.15, 0.225, 0.30)[0]) == 0.0
    assert float(_tri(np.array([0.225]), 0.15, 0.225, 0.30)[0]) == 1.0
    assert float(_trap_hi(np.array([0.10]), 0.30, 0.45)[0]) == 0.0
    assert float(_trap_hi(np.array([0.45]), 0.30, 0.45)[0]) == 1.0
    assert float(_trap_hi(np.array([99.0]), 0.30, 0.45)[0]) == 1.0


def test_critical_boundary_at_threshold() -> None:
    diag = diagnose(_pred(v_b=0.45, ra=0.4), thresholds=FuzzyThresholds())
    assert diag.state == "critical"
    assert diag.wear_membership == 1.0


def test_explanation_names_each_driver() -> None:
    wear = diagnose(_pred(v_b=0.6, ra=0.4), thresholds=FuzzyThresholds())
    assert "flank wear" in wear.explanation
    rough = diagnose(_pred(v_b=0.05, ra=3.0), thresholds=FuzzyThresholds())
    assert "roughness" in rough.explanation
    assert not state_is_safe(wear) and not state_is_safe(rough)
    calm = diagnose(_pred(v_b=0.05, ra=0.4), thresholds=FuzzyThresholds())
    assert "safe envelope" in calm.explanation


def test_thresholds_scale_with_target_finish() -> None:
    tight = FuzzyThresholds.for_target(0.8)
    assert (tight.ra_low, tight.ra_mid, tight.ra_high) == approx((0.4, 0.8, 1.2))
    assert FuzzyThresholds.for_target(0.0) == FuzzyThresholds()


def test_same_roughness_judged_against_tolerance() -> None:
    pred = _pred(v_b=0.05, ra=1.2)
    assert diagnose(pred, target_ra_um=0.8).state == "critical"
    assert diagnose(pred, target_ra_um=3.2).state == "safe"


def test_explicit_thresholds_win_over_target() -> None:
    pred = _pred(v_b=0.05, ra=1.2)
    diag = diagnose(pred, thresholds=FuzzyThresholds(), target_ra_um=0.8)
    assert diag.state == "caution"


def test_clipped_extremes_stay_critical() -> None:
    from dataclasses import replace

    assert diagnose(_pred(v_b=1.5, ra=0.4)).state == "critical"
    assert diagnose(_pred(v_b=0.05, ra=6.0)).state == "critical"
    vib = generate_synthetic_sensor(v_c=300.0, f=0.4, ap=3.0, seed=11)
    vib = replace(vib, vibration_rms_g=99.0)
    diag = diagnose(_pred(v_b=0.05, ra=0.4), sensor=vib)
    assert diag.state == "critical"
