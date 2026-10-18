from __future__ import annotations

from hybrid_ai.stages.ann.model import TrainConfig, WearRoughnessModel
from hybrid_ai.core.schemas import ToolCandidate
from hybrid_ai.core.synthetic import generate_synthetic_dataset
from hybrid_ai.stages.fuzzy.diagnoser import diagnose
from hybrid_ai.stages.nsga2.optimiser import (
    NsgaConfig,
    mrr_cm3_per_min,
    optimise,
    tool_life_min,
)


def _fitted_model() -> WearRoughnessModel:
    x, y = generate_synthetic_dataset(n_samples=400, seed=3)
    model = WearRoughnessModel()
    model.fit(x, y, TrainConfig(epochs=60, hidden=(32, 32)))
    return model


def _tool() -> ToolCandidate:
    return ToolCandidate(
        insert_code="CNMG120408-PM",
        iso_group="P",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=180.0,
        v_c_max_mpm=320.0,
        f_min_mmpr=0.15,
        f_max_mmpr=0.45,
        ap_max_mm=4.0,
    )


def test_mrr_increases_with_speed_and_feed() -> None:
    from hybrid_ai.core.schemas import OperatingPoint

    low = mrr_cm3_per_min(OperatingPoint(v_c_mpm=150.0, f_mmpr=0.2, ap_mm=1.5))
    high = mrr_cm3_per_min(OperatingPoint(v_c_mpm=250.0, f_mmpr=0.3, ap_mm=1.5))
    assert high > low


def test_mrr_exact_value() -> None:
    from hybrid_ai.core.schemas import OperatingPoint

    mrr = mrr_cm3_per_min(OperatingPoint(v_c_mpm=200.0, f_mmpr=0.3, ap_mm=2.0))
    assert mrr == 2.0 * 0.3 * 200.0 / 1000.0 == 0.12


def test_tool_life_clip_bounds() -> None:
    assert tool_life_min(_op(v_c=200, f=0.3, ap=2.0), v_b=0.0) == 240.0
    assert tool_life_min(_op(v_c=200, f=0.3, ap=2.0), v_b=100.0) == 1.0


def test_tool_life_decreases_with_higher_wear() -> None:
    a = tool_life_min(_op(v_c=150, f=0.2, ap=1.5), v_b=0.05)
    b = tool_life_min(_op(v_c=250, f=0.4, ap=2.0), v_b=0.5)
    assert a > b


def test_tool_life_in_realistic_minutes() -> None:
    assert tool_life_min(_op(v_c=200, f=0.3, ap=2.0), v_b=0.3) == 20.0
    assert 1.0 <= tool_life_min(_op(v_c=200, f=0.3, ap=2.0), v_b=0.02) <= 240.0


def _op(v_c: float, f: float, ap: float):
    from hybrid_ai.core.schemas import OperatingPoint
    return OperatingPoint(v_c_mpm=v_c, f_mmpr=f, ap_mm=ap)


def test_optimise_returns_pareto_front() -> None:
    model = _fitted_model()
    front = optimise(model, _tool(), config=NsgaConfig(pop_size=20, n_generations=10))
    assert len(front) >= 1
    for s in front:
        assert s.mrr_cm3_per_min > 0
        assert s.tool_life_min > 0
        assert s.ra_um > 0


def test_pareto_front_is_non_dominated() -> None:
    model = _fitted_model()
    front = optimise(model, _tool(), config=NsgaConfig(pop_size=20, n_generations=10))
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            if i == j:
                continue
            better_or_equal = (
                b.mrr_cm3_per_min >= a.mrr_cm3_per_min
                and b.tool_life_min >= a.tool_life_min
                and b.ra_um <= a.ra_um
            )
            strictly_better = (
                b.mrr_cm3_per_min > a.mrr_cm3_per_min
                or b.tool_life_min > a.tool_life_min
                or b.ra_um < a.ra_um
            )
            assert not (better_or_equal and strictly_better), (
                f"Solution {i} dominated by {j}"
            )


def test_enforce_safe_returns_only_safe_points() -> None:
    model = _fitted_model()
    front = optimise(
        model, _tool(),
        config=NsgaConfig(pop_size=20, n_generations=10, enforce_safe=True),
        target_ra_um=1.6,
    )
    assert len(front) >= 1
    for s in front:
        pred = _pred_for(model, s.point)
        assert diagnose(pred, None, target_ra_um=1.6).state == "safe"


def _pred_for(model, point):
    from hybrid_ai.stages.ann.model import predict_for_point

    return predict_for_point(model, _tool(), point, hardness_hb=200.0)


def test_optimise_is_deterministic_for_fixed_seed() -> None:
    model = _fitted_model()
    cfg = NsgaConfig(pop_size=20, n_generations=10, seed=7)
    first = [s.to_dict() for s in optimise(model, _tool(), config=cfg)]
    second = [s.to_dict() for s in optimise(model, _tool(), config=cfg)]
    assert first == second


def test_pareto_filter_units() -> None:
    from hybrid_ai.core.schemas import OperatingPoint, ParetoSolution
    from hybrid_ai.stages.nsga2.optimiser import _pareto_filter

    def _sol(vc, mrr, life, ra):
        return ParetoSolution(point=OperatingPoint(vc, 0.2, 1.0),
                              mrr_cm3_per_min=mrr, tool_life_min=life, ra_um=ra)

    assert _pareto_filter([]) == []
    lone = _sol(200, 1.0, 10.0, 1.0)
    assert _pareto_filter([lone]) == [lone]
    dom = _sol(200, 1.0, 10.0, 1.0)
    better = _sol(250, 2.0, 20.0, 0.5)
    assert _pareto_filter([dom, better]) == [better]
    dup = _sol(200, 1.0, 10.0, 1.0)
    assert len(_pareto_filter([lone, dup])) == 1


def test_cached_and_uncached_agree() -> None:
    from hybrid_ai.stages.nsga2 import optimiser as opt

    model = _fitted_model()
    tool = _tool()
    point = _op(v_c=200, f=0.3, ap=2.0)
    with_cache, _ = opt._predict_cached(model, tool, point, 200.0, None, {})
    without, _ = opt._predict_cached(model, tool, point, 200.0, None, None)
    assert (with_cache.v_b_mm, with_cache.ra_um) == (without.v_b_mm, without.ra_um)


def test_prediction_cache_avoids_recompute() -> None:
    from hybrid_ai.stages.nsga2 import optimiser as opt

    model = _fitted_model()
    tool = _tool()
    calls = {"n": 0}
    real_predict = opt.predict_for_point

    def counting(model_, tool_, point, **kwargs):
        calls["n"] += 1
        return real_predict(model_, tool_, point, **kwargs)

    opt.predict_for_point = counting
    try:
        point = _op(v_c=200, f=0.3, ap=2.0)
        cache: dict = {}
        opt._predict_cached(model, tool, point, 200.0, None, cache)
        opt._predict_cached(model, tool, point, 200.0, None, cache)
        assert calls["n"] == 1
        assert len(cache) == 1
    finally:
        opt.predict_for_point = real_predict


def test_grid_fallback_respects_safe_gate(monkeypatch) -> None:
    from hybrid_ai.stages.nsga2 import optimiser as opt

    monkeypatch.setattr(opt, "_HAS_PYMOO", False)
    model = _fitted_model()
    front = optimise(
        model, _tool(),
        config=NsgaConfig(pop_size=20, n_generations=10, enforce_safe=True),
        target_ra_um=1.6,
    )
    assert len(front) >= 1
    for s in front:
        pred = _pred_for(model, s.point)
        assert diagnose(pred, None, target_ra_um=1.6).state == "safe"
