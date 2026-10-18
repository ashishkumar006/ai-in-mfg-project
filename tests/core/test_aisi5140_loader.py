from __future__ import annotations

import numpy as np

from hybrid_ai.core.aisi5140_loader import (
    AISI5140_HARDNESS_HB,
    load_aisi5140,
)


def test_shapes_and_units() -> None:
    x, y = load_aisi5140()
    assert x.shape == (9, 11)
    assert y.shape == (9, 2)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert 0.05 < y[:, 0].max() < 1.0  # VB in mm
    assert 0.2 < y[:, 1].min() and y[:, 1].max() < 3.0  # Ra in um, measured


def test_feature_schema() -> None:
    x, _ = load_aisi5140()
    assert np.all(x[:, 3] == 0.8)  # nose radius
    assert np.all(x[:, 4] == 0.0)  # rake
    assert np.all(x[:, 5] == 1.0)  # ISO P one-hot
    assert np.all(x[:, 6:10] == 0.0)
    assert np.all(x[:, 10] == AISI5140_HARDNESS_HB)
    assert set(np.unique(x[:, 0]).tolist()) == {150.0, 200.0, 300.0}
