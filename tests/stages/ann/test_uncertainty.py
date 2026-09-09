from __future__ import annotations

import numpy as np

from hybrid_ai.stages.ann.model import (
    DLConfig,
    WearRoughnessDL,
    predict_for_point,
)
from hybrid_ai.core.schemas import OperatingPoint, ToolCandidate


def _tiny_dl(dropout: float = 0.5) -> WearRoughnessDL:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 2, 32))
    y = rng.normal(size=(64, 2))
    model = WearRoughnessDL(config=DLConfig(
        model_type="cnn1d", epochs=2, batch_size=32,
        n_channels=2, window_len=32, dropout=dropout))
    model.fit(x, y)
    return model


def test_mc_returns_mean_and_spread() -> None:
    model = _tiny_dl()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 2, 32))
    mean, std = model.predict_with_uncertainty(x, n_passes=10)
    assert mean.shape == (3, 2) and std.shape == (3, 2)
    assert np.all(std >= 0.0)


def test_mc_is_seeded() -> None:
    model = _tiny_dl()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 32))
    m1, s1 = model.predict_with_uncertainty(x, n_passes=8, seed=7)
    m2, s2 = model.predict_with_uncertainty(x, n_passes=8, seed=7)
    assert np.allclose(m1, m2) and np.allclose(s1, s2)


def test_no_dropout_means_no_spread() -> None:
    model = _tiny_dl(dropout=0.0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 32))
    _, std = model.predict_with_uncertainty(x, n_passes=8)
    assert np.allclose(std, 0.0)


def test_predict_for_point_reports_uncertainty() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 1, 11))
    y = rng.normal(size=(64, 2))
    model = WearRoughnessDL(config=DLConfig(
        model_type="cnn1d", epochs=2, batch_size=32,
        n_channels=1, window_len=11, dropout=0.5))
    model.fit(x, y)
    tool = ToolCandidate(
        insert_code="T", iso_group="P", operation="turning",
        nose_radius_mm=0.8, rake_angle_deg=6.0,
        v_c_min_mpm=100.0, v_c_max_mpm=200.0,
        f_min_mmpr=0.1, f_max_mmpr=0.4, ap_max_mm=3.0,
    )
    point = OperatingPoint(v_c_mpm=150.0, f_mmpr=0.2, ap_mm=1.5)
    pred = predict_for_point(model, tool, point, n_mc=10)
    assert pred.uncertainty >= 0.0
    assert pred.to_dict()["uncertainty"] == pred.uncertainty
