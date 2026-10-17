from __future__ import annotations

import pytest

from hybrid_ai.core.schemas import PartSpec
from hybrid_ai.stages.clips.rules import build_capp_sheet
from hybrid_ai.tools import _models, replan


def _part(**kw) -> PartSpec:
    d = dict(part_id="X", material="m", iso_group="P", operation="turning",
             machine="medium", hardness_hb=220.0, target_ra_um=1.6,
             batch_size=10)
    d.update(kw)
    return PartSpec(**d)


def test_replan_resolves_when_a_safer_setting_exists() -> None:
    """A hard P part whose envelope midpoint is critical: the loop must be
    able to find a safe insert+setting combination."""
    part = _part(hardness_hb=300.0, target_ra_um=0.8)
    _, _, sel, _, _ = build_capp_sheet(part, _models.get_catalog())
    out = replan.run(part, failed_insert=sel.insert_code, failed_state="critical")
    assert out["resolved"] is True
    assert out["reading"] == "safe"
    alt = out["alternative"]
    assert alt["diagnosis"]["state"] == "safe"
    assert alt["insert_code"] != sel.insert_code
    vc = alt["recommended_point"]["v_c_mpm"]
    assert alt["envelope"]["v_c_mpm"][0] <= vc <= alt["envelope"]["v_c_mpm"][1]


def test_replan_reports_no_alternative_honestly(monkeypatch) -> None:
    """Where the catalog has no other insert to try, the loop must say
    so, not invent a fix."""
    mini = [t for t in _models.get_catalog()
            if t.insert_code == "SDMX090308-P"]
    assert mini, "driving insert must exist in the bundled catalog"
    monkeypatch.setattr(_models, "get_catalog", lambda: list(mini))
    part = _part(iso_group="K", operation="drilling", machine="heavy",
                 hardness_hb=330.0, target_ra_um=0.4, batch_size=5)
    out = replan.run(part, failed_insert="SDMX090308-P", failed_state="critical")
    assert out["resolved"] is False
    assert out["reading"] == "no_alternative"
    assert out["alternative"] is None


def test_replan_unknown_insert_is_a_no_op_not_a_crash() -> None:
    """Excluding an insert that is not in the catalog must still return a
    valid plan (the catalog is unchanged)."""
    part = _part()
    out = replan.run(part, failed_insert="NOT-A-REAL-CODE")
    assert out["alternative"] is not None
    assert out["failed_insert"] == "NOT-A-REAL-CODE"
    assert set(out["considered"]) == {t.insert_code
                                      for t in _models.get_catalog()}


def test_replan_considered_excludes_the_failed_insert() -> None:
    part = _part()
    _, _, sel, _, _ = build_capp_sheet(part, _models.get_catalog())
    out = replan.run(part, failed_insert=sel.insert_code)
    assert sel.insert_code not in out["considered"]


@pytest.mark.parametrize("bounds,n", [
    ([160.0, 280.0], 3), ([160.0, 280.0], 1), ([200.0, 200.0], 3),
])
def test_linspace_stays_in_bounds(bounds, n) -> None:
    pts = replan._linspace(bounds, n)
    assert all(bounds[0] <= p <= bounds[1] for p in pts)
    assert pts
