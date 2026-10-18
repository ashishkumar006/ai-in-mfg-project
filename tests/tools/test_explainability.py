from __future__ import annotations

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import compare_models, optimize_pareto


def _part(**kwargs) -> PartSpec:
    defaults = dict(
        part_id="X", material="m", iso_group="K", operation="turning",
        machine="light", hardness_hb=330.0, target_ra_um=0.4, batch_size=10,
    )
    defaults.update(kwargs)
    return PartSpec(**defaults)


IMPOSSIBLE = _part()


def test_impossible_front_explains_itself() -> None:
    body = optimize_pareto.run(IMPOSSIBLE, insert_code="CNMG120408-KC",
                              pop_size=12, n_generations=5, safe_only=True)
    assert body["solutions"] == []
    assert body["extremes"] == {}
    info = body["no_safe_front"]
    assert info["reason"] == "no_safe_setting_exists"
    assert info["diagnosis"]["state"] == "critical"
    assert info["diagnosis"]["explanation"]
    assert info["diagnosis"]["v_b_mm"] > 0
    assert info["envelope"]["v_c_mpm"][0] < info["envelope"]["v_c_mpm"][1]
    assert info["target_ra_um"] == 0.4
    assert "safe-only" in info["hint"]


def test_relaxing_finish_recovers_a_front() -> None:
    relaxed = _part(target_ra_um=3.2)
    body = optimize_pareto.run(relaxed, insert_code="CNMG120408-KC",
                              pop_size=20, n_generations=8, safe_only=True)
    assert body["solutions"], "a loose finish must yield safe settings"
    assert "no_safe_front" not in body


def test_unsafemode_still_returns_points() -> None:
    body = optimize_pareto.run(IMPOSSIBLE, insert_code="CNMG120408-KC",
                              pop_size=12, n_generations=5, safe_only=False)
    assert body["solutions"], "gate off must always return the trade-off space"
    assert "no_safe_front" not in body


def test_compare_models_tiers_rows_and_reports_spread() -> None:
    point = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.15, ap_mm=1.0)
    body = compare_models.run(IMPOSSIBLE, point,
                              insert_code="CNMG120408-KC")
    tiers = {r["tier"] for r in body["rows"]}
    assert tiers == {"production", "reference"}
    prod = [r for r in body["rows"] if r["tier"] == "production"]
    ref = [r for r in body["rows"] if r["tier"] == "reference"]
    assert prod, "at least one trusted model"
    # the two 145-row models are the reference tier
    assert {r["model"] for r in ref} == {"CNN1D", "CNN-LSTM"}
    for r in ref:
        assert "overfit" in (r.get("caveat") or "")
    spread = body["production_spread_mm"]
    assert spread is not None
    vbs = [r["v_b_mm"] for r in prod if r["v_b_mm"] is not None]
    assert abs(spread - (max(vbs) - min(vbs))) < 1e-6
    for r in body["rows"]:
        assert r.get("basis") in {"point", "reference-window"}, r["model"]
    for r in body["rows"]:
        if r.get("basis") == "reference-window" and r.get("v_b_mm") is not None:
            assert r.get("true_v_b_mm") is not None, r["model"]
            assert r.get("vb_error_mm") is not None, r["model"]
            assert abs(r["vb_error_mm"] - (r["v_b_mm"] - r["true_v_b_mm"])) < 0.002


def test_reference_models_excluded_from_spread() -> None:
    point = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.15, ap_mm=1.0)
    body = compare_models.run(IMPOSSIBLE, point,
                              insert_code="CNMG120408-KC")
    all_vb = [r["v_b_mm"] for r in body["rows"] if r["v_b_mm"] is not None]
    prod_vb = [r["v_b_mm"] for r in body["rows"]
               if r["tier"] == "production" and r["v_b_mm"] is not None]
    all_spread = max(all_vb) - min(all_vb)
    assert body["production_spread_mm"] < all_spread, (
        "the reference tier is what inflates the disagreement; the "
        "production-only figure must be smaller"
    )