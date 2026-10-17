from __future__ import annotations

import pytest

from hybrid_ai.core.schemas import OperatingPoint
from hybrid_ai.stages.nsga2.taylor import (
    TAYLOR_TABLE,
    compare_life,
    taylor_constant,
    taylor_tool_life_min,
)


def _pt(v=200.0, f=0.3, ap=1.5) -> OperatingPoint:
    return OperatingPoint(v_c_mpm=v, f_mmpr=f, ap_mm=ap)


def test_constant_matches_reference_point() -> None:
    for iso, (n, v0, t0) in TAYLOR_TABLE.items():
        assert taylor_constant(iso) == pytest.approx(v0 * t0 ** n)


def test_reference_point_gives_reference_life() -> None:
    """At the reference speed/feed/depth the equation must return T0."""
    for iso, (_, v0, t0) in TAYLOR_TABLE.items():
        r = taylor_tool_life_min(_pt(v=v0, f=1.0, ap=1.0), iso)
        assert r.tool_life_min == pytest.approx(t0, rel=1e-6)


def test_life_falls_with_speed_and_feed() -> None:
    slow = taylor_tool_life_min(_pt(v=150.0), "P")
    fast = taylor_tool_life_min(_pt(v=300.0), "P")
    assert fast.tool_life_min < slow.tool_life_min
    light_feed = taylor_tool_life_min(_pt(f=0.15), "P")
    heavy_feed = taylor_tool_life_min(_pt(f=0.45), "P")
    assert heavy_feed.tool_life_min < light_feed.tool_life_min


def test_speed_dominates_feed() -> None:
    """Effective life exponents are -1/n (speed, ~-4) and -F/n (feed, ~-0.8),
    so doubling the speed must shorten life far more than doubling the feed.
    """
    base = taylor_tool_life_min(_pt(), "P").tool_life_min
    dbl_speed = taylor_tool_life_min(_pt(v=400.0), "P").tool_life_min
    dbl_feed = taylor_tool_life_min(_pt(f=0.6), "P").tool_life_min
    assert dbl_speed < dbl_feed < base


def test_life_is_clipped_to_bounds() -> None:
    tiny = taylor_tool_life_min(_pt(v=1e6, f=10.0, ap=10.0), "P")
    assert tiny.tool_life_min >= 1.0
    huge = taylor_tool_life_min(_pt(v=1e-6, f=1e-6, ap=1e-6), "P")
    assert huge.tool_life_min <= 240.0


def test_unknown_group_raises() -> None:
    with pytest.raises(KeyError):
        taylor_tool_life_min(_pt(), "Z")


def test_compare_life_verdicts() -> None:
    point = _pt(v=200.0, f=0.3, ap=1.5)
    taylor_life = taylor_tool_life_min(point, "P").tool_life_min
    same = compare_life(0.3, taylor_life, point, "P")
    assert same["verdict"] == "consistent"
    optimistic = compare_life(0.3, taylor_life / 4.0, point, "P")
    assert optimistic["verdict"] == "ann_optimistic"
    pessimistic = compare_life(0.3, taylor_life * 4.0, point, "P")
    assert pessimistic["verdict"] == "ann_pessimistic"


def test_compare_life_payload_round_trips() -> None:
    out = compare_life(0.2, 30.0, _pt(), "M")
    for key in ("ann_life_min", "taylor_life_min", "ratio_taylor_over_ann",
                "verdict", "reading", "point"):
        assert key in out
    assert out["point"]["v_c_mpm"] == 200.0
