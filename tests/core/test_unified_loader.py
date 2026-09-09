from __future__ import annotations

import numpy as np

from hybrid_ai.core.unified_loader import TEST_GROUPS, TRAIN_GROUPS, VAL_GROUPS, load_unified_windows


def test_pool_shapes_and_rate() -> None:
    x, y, groups = load_unified_windows(split="train")
    assert x.ndim == 3 and x.shape[1:] == (3, 256)
    assert y.shape == (x.shape[0], 2)
    assert x.dtype == np.float32
    assert set(groups.tolist()) == set(TRAIN_GROUPS)


def test_pool_splits_do_not_share_experiments() -> None:
    tr = set(load_unified_windows(split="train")[2].tolist())
    va = set(load_unified_windows(split="val")[2].tolist())
    te = set(load_unified_windows(split="test")[2].tolist())
    assert va == set(VAL_GROUPS)
    assert te == set(TEST_GROUPS)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_pool_covers_both_families() -> None:
    _, _, groups = load_unified_windows()
    tags = set(groups.tolist())
    assert any(t.startswith("phm:") for t in tags)
    assert any(t.startswith("nuaa:") for t in tags)
