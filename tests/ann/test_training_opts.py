from __future__ import annotations

import numpy as np

from hybrid_ai.ann.model import DLConfig, WearRoughnessDL
from hybrid_ai.ann.vicomtech_model import VicomtechConfig, VicomtechTurningModel
from hybrid_ai.core.vicomtech_loader import load_vicomtech


def _tiny_windows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(160, 4, 64))
    y = rng.normal(size=(160, 2))
    return x, y


def test_early_stopping_halts_and_restores_best() -> None:
    x, y = _tiny_windows()
    cfg = DLConfig(model_type="cnn1d", epochs=30, batch_size=32,
                   n_channels=4, window_len=64, patience=3)
    model = WearRoughnessDL(config=cfg)
    history = model.fit(x[:128], y[:128], val_data=(x[128:], y[128:]))
    assert history["stopped_early"] is True
    assert history["epochs_run"] < 30
    assert np.isfinite(history["best_val_loss"])


def test_weight_decay_runs() -> None:
    x, y = _tiny_windows()
    cfg = DLConfig(model_type="cnn1d", epochs=2, batch_size=32,
                   n_channels=4, window_len=64, weight_decay=1e-4)
    model = WearRoughnessDL(config=cfg)
    history = model.fit(x, y)
    assert history["epochs_run"] == 2
    assert model.predict(x[:2]).shape == (2, 2)


def test_vicomtech_model_roundtrip(tmp_path) -> None:
    x, y, _ = load_vicomtech(split="train")
    model = VicomtechTurningModel(config=VicomtechConfig(max_iter=10))
    model.fit(x[:200], y[:200])
    assert model.predict(x[:4]).shape == (4, 2)
    ckpt = tmp_path / "turning.joblib"
    model.save(ckpt)
    back = VicomtechTurningModel.load(ckpt)
    assert np.allclose(back.predict(x[:4]), model.predict(x[:4]))
    metrics = back.evaluate(x[:50], y[:50])
    assert metrics["n"] == 50 and metrics["rmse"] > 0
