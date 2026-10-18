from __future__ import annotations

import numpy as np
import pytest

from hybrid_ai.core.luh_loader import LUH_H5_DIR, load_luh_features
from hybrid_ai.stages.ann.luh_model import LUHConfig, LUHMillingModel

pytestmark = pytest.mark.skipif(
    not (LUH_H5_DIR.exists() and any(LUH_H5_DIR.glob("*.h5"))),
    reason="LUH h5 subset not downloaded (scripts/checks/download_luh_subset.py)",
)


def _small_batch():
    x16, vb_mm, contact_s, _tools, _runs = load_luh_features()
    x = LUHMillingModel.assemble(x16[:200], contact_s[:200])
    return x, np.asarray(vb_mm[:200], dtype=np.float64)


def test_assemble_stacks_log_contact_time() -> None:
    x16 = np.arange(32, dtype=np.float64).reshape(2, 16)
    contact = np.array([0.0, 3.0])
    x = LUHMillingModel.assemble(x16, contact)
    assert x.shape == (2, 17)
    assert np.allclose(x[:, :-1], x16)
    assert np.allclose(x[:, -1], np.log1p(contact))


def test_single_target_roundtrip(tmp_path) -> None:
    x, vb = _small_batch()
    model = LUHMillingModel(config=LUHConfig(max_iter=10))
    model.fit(x, vb)
    assert model.predict(x[:4]).shape == (4,)

    ckpt = tmp_path / "luh_hgbr.joblib"
    model.save(ckpt)
    back = LUHMillingModel.load(ckpt)
    assert np.allclose(back.predict(x[:4]), model.predict(x[:4]))

    metrics = back.evaluate(x[:50], vb[:50])
    assert metrics["n"] == 50 and metrics["rmse"] > 0
