from __future__ import annotations

import numpy as np
import pytest

from hybrid_ai.core.luh_loader import (
    FEATURE_NAMES,
    LUH_H5_DIR,
    LUH_TOOL_MACHINE,
    featurise_file,
    load_luh_features,
)

pytestmark = pytest.mark.skipif(
    not (LUH_H5_DIR.exists() and any(LUH_H5_DIR.glob("*.h5"))),
    reason="LUH h5 subset not downloaded (scripts/checks/download_luh_subset.py)",
)


def test_shapes_units_and_alignment() -> None:
    x, vb_mm, contact_s, tools, runs = load_luh_features()
    assert x.shape == (len(vb_mm), len(FEATURE_NAMES))
    assert x.shape[0] == len(contact_s) == len(tools) == len(runs)
    assert np.all(np.isfinite(x))
    assert 0.0 < float(vb_mm.min()) and float(vb_mm.max()) < 0.25
    assert set(np.unique(tools).tolist()) <= set(LUH_TOOL_MACHINE)


def test_contact_time_increases_with_run_per_tool() -> None:
    _, _, contact_s, tools, runs = load_luh_features()
    for tool in np.unique(tools):
        mask = tools == tool
        order = np.argsort(runs[mask])
        assert np.all(np.diff(contact_s[mask][order]) >= 0)


def test_featurise_is_deterministic() -> None:
    sample = sorted(LUH_H5_DIR.glob("*.h5"))[0]
    a = featurise_file(sample)
    b = featurise_file(sample)
    assert a.shape == (len(FEATURE_NAMES),)
    assert np.array_equal(a, b)


def test_cache_returns_identical_arrays() -> None:
    first = load_luh_features()
    second = load_luh_features()
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
