from __future__ import annotations

import numpy as np

from hybrid_ai.core.haas_loader import (
    HAAS_TEST_TOOLS,
    HAAS_TRAIN_TOOLS,
    HAAS_VAL_TOOLS,
    feature_columns,
    load_haas,
    load_haas_sequences,
)


def test_shapes_and_no_nans() -> None:
    x, y, groups = load_haas()
    assert x.shape == (968, len(feature_columns()))
    assert y.shape == (968,)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    assert "HardnessMean" in feature_columns()


def test_splits_cover_all_tools_without_leakage() -> None:
    tr = set(load_haas(split="train")[2].tolist())
    va = set(load_haas(split="val")[2].tolist())
    te = set(load_haas(split="test")[2].tolist())
    assert tr == set(HAAS_TRAIN_TOOLS)
    assert va == set(HAAS_VAL_TOOLS)
    assert te == set(HAAS_TEST_TOOLS)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_sequences_follow_cycle_order() -> None:
    seqs = load_haas_sequences(split="test")
    assert {t for _, _, t in seqs} == set(HAAS_TEST_TOOLS)
    for x, y, _ in seqs:
        assert x.shape[0] == y.shape[0] > 0
        assert y[0] >= y[-1]  # remaining life decreases along a tool's life
