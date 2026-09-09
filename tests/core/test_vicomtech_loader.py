from __future__ import annotations

import numpy as np

from hybrid_ai.core.vicomtech_loader import (
    VICOMTECH_TEST_TOOLS,
    VICOMTECH_TRAIN_TOOLS,
    VICOMTECH_VAL_TOOLS,
    feature_columns,
    load_vicomtech,
)


def test_shapes_and_units() -> None:
    x, y, groups = load_vicomtech()
    assert x.shape == (2054, len(feature_columns()))
    assert y.shape == (2054, 2)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert y[:, 0].max() < 1.0  # Vb converted um -> mm


def test_splits_cover_all_tools_without_leakage() -> None:
    tr = set(load_vicomtech(split="train")[2].tolist())
    va = set(load_vicomtech(split="val")[2].tolist())
    te = set(load_vicomtech(split="test")[2].tolist())
    assert tr == set(VICOMTECH_TRAIN_TOOLS)
    assert va == set(VICOMTECH_VAL_TOOLS)
    assert te == set(VICOMTECH_TEST_TOOLS)
    assert not (tr & va) and not (tr & te) and not (va & te)
