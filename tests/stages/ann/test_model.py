from __future__ import annotations

import numpy as np

from hybrid_ai.stages.ann.model import (
    DLConfig,
    SklearnBaseline,
    TrainConfig,
    WearRoughnessDL,
    WearRoughnessModel,
    fine_tune_head,
    freeze_backbone,
    predict_for_point,
    unfreeze_all,
)
from hybrid_ai.core.schemas import OperatingPoint, ToolCandidate
from hybrid_ai.core.synthetic import generate_synthetic_dataset


def _small_model() -> WearRoughnessModel:
    x, y = generate_synthetic_dataset(n_samples=400, seed=1)
    model = WearRoughnessModel()
    model.fit(x, y, TrainConfig(epochs=20, hidden=(16, 16)))
    return model


def test_model_trains_and_predicts_shape() -> None:
    x, y = generate_synthetic_dataset(n_samples=200, seed=0)
    model = WearRoughnessModel()
    history = model.fit(x, y, TrainConfig(epochs=10, hidden=(16,)))
    assert "loss_history" in history
    preds = model.predict(x[:5])
    assert preds.shape == (5, 2)


def test_model_beats_constant_baseline_on_holdout() -> None:
    x, y = generate_synthetic_dataset(n_samples=600, seed=2)
    split = int(0.8 * x.shape[0])
    model = WearRoughnessModel()
    model.fit(x[:split], y[:split], TrainConfig(epochs=30, hidden=(32, 32)))
    metrics = model.evaluate(x[split:], y[split:])
    baseline = float(np.mean((y[split:] - y[:split].mean(axis=0)) ** 2)) ** 0.5
    assert metrics["rmse"] < baseline


def test_predict_for_point_returns_positive_values() -> None:
    model = _small_model()
    tool = ToolCandidate(
        insert_code="CNMG120408-PM",
        iso_group="P",
        operation="turning",
        nose_radius_mm=0.8,
        rake_angle_deg=6.0,
        v_c_min_mpm=180.0,
        v_c_max_mpm=320.0,
        f_min_mmpr=0.15,
        f_max_mmpr=0.45,
        ap_max_mm=4.0,
    )
    point = OperatingPoint(v_c_mpm=240.0, f_mmpr=0.25, ap_mm=2.0)
    pred = predict_for_point(model, tool, point, hardness_hb=200.0)
    assert pred.v_b_mm > 0
    assert pred.ra_um > 0


def test_sklearn_baseline_runs() -> None:
    x, y = generate_synthetic_dataset(n_samples=300, seed=4)
    baseline = SklearnBaseline()
    baseline.fit(x, y)
    preds = baseline.predict(x[:5])
    assert preds.shape == (5, 2)
    assert (preds > 0).all()


def test_cnn1d_trains_on_windows() -> None:
    x, y = generate_synthetic_dataset(n_samples=300, seed=7)
    x_windows = np.stack([row[np.newaxis, :] for row in x])
    cfg = DLConfig(model_type="cnn1d", epochs=5, n_channels=1, window_len=x_windows.shape[2])
    model = WearRoughnessDL(config=cfg)
    history = model.fit(x_windows, y)
    assert "loss_history" in history
    preds = model.predict(x_windows[:3])
    assert preds.shape == (3, 2)


def test_cnn_lstm_trains_on_windows() -> None:
    x, y = generate_synthetic_dataset(n_samples=300, seed=8)
    x_windows = np.stack([row[np.newaxis, :] for row in x])
    cfg = DLConfig(
        model_type="cnn_lstm",
        epochs=5,
        n_channels=1,
        window_len=x_windows.shape[2],
        lstm_hidden=32,
        lstm_layers=1,
    )
    model = WearRoughnessDL(config=cfg)
    history = model.fit(x_windows, y)
    assert "loss_history" in history
    preds = model.predict(x_windows[:3])
    assert preds.shape == (3, 2)


def test_dl_config_defaults() -> None:
    cfg = DLConfig()
    assert cfg.model_type == "cnn1d"
    assert cfg.n_channels == 6
    assert cfg.window_len == 9000


def test_freeze_unfreeze_backbone() -> None:
    x, y = generate_synthetic_dataset(n_samples=100, seed=10)
    x_windows = np.stack([row[np.newaxis, :] for row in x])
    cfg = DLConfig(model_type="cnn_lstm", epochs=2, n_channels=1, window_len=x_windows.shape[2])
    model = WearRoughnessDL(config=cfg)
    model.fit(x_windows, y)
    freeze_backbone(model)
    head_grad = all(p.requires_grad for p in model.model.head.parameters())
    backbone_grad = all(p.requires_grad for p in model.model.features.parameters())
    assert head_grad
    assert not backbone_grad
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.model.parameters())


def test_fine_tune_head_roundtrip(tmp_path) -> None:
    x, y = generate_synthetic_dataset(n_samples=100, seed=11)
    x_windows = np.stack([row[np.newaxis, :] for row in x])
    cfg = DLConfig(
        model_type="cnn1d", epochs=2, n_channels=1, window_len=x_windows.shape[2]
    )
    pretrained = WearRoughnessDL(config=cfg)
    pretrained.fit(x_windows, y)
    ckpt = tmp_path / "pretrained.pt"
    pretrained.save(ckpt)

    x_target, y_target = generate_synthetic_dataset(n_samples=50, seed=12)
    x_target_windows = np.stack([row[np.newaxis, :] for row in x_target])
    fine_tuned = fine_tune_head(ckpt, x_target_windows, y_target, epochs=3)
    preds = fine_tuned.predict(x_target_windows[:3])
    assert preds.shape == (3, 2)
