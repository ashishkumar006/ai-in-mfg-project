from __future__ import annotations

import numpy as np

from hybrid_ai.stages.ann.model import DLConfig, WearRoughnessDL, freeze_backbone
from hybrid_ai.core.phm_loader import load_phm_windows


def test_phm_windows_shapes_and_groups() -> None:
    x, y, groups = load_phm_windows(window=256, stride=16)
    assert x.ndim == 3 and x.shape[1:] == (7, 256)
    assert y.shape == (x.shape[0], 2)
    assert set(groups.tolist()) == {"c1", "c4", "c6"}
    assert np.all(y[:, 0] >= 0.0)


def test_phm_split_has_no_cutter_leakage() -> None:
    _, _, gtr = load_phm_windows(window=256, stride=16, split="train")
    _, _, gte = load_phm_windows(window=256, stride=16, split="test")
    assert set(gte.tolist()) == {"c6"}
    assert not (set(gtr.tolist()) & set(gte.tolist()))


def test_phm_labels_are_causal() -> None:
    import pandas as pd

    from hybrid_ai.core.phm_loader import PHM_CHANNELS, PHM_PATH

    df = pd.read_csv(PHM_PATH, usecols=["experiment_tag", "tool_wear"])
    x, y, groups = load_phm_windows(window=64, stride=64, use_cache=False)
    first_c1 = df[df["experiment_tag"] == "c1"].reset_index(drop=True)
    assert float(first_c1.loc[63, "tool_wear"]) == y[groups == "c1"][0, 0]


def test_cnn_bilstm_trains_and_roundtrips(tmp_path) -> None:
    x, y, _ = load_phm_windows(window=256, stride=16, split="train")
    cfg = DLConfig(model_type="cnn_bilstm", epochs=1, batch_size=64,
                   n_channels=7, window_len=256, lstm_hidden=32,
                   lstm_layers=1)
    model = WearRoughnessDL(config=cfg)
    history = model.fit(x[:128], y[:128])
    assert "final_loss" in history
    model.model.eval()
    preds = model.predict(x[:4])
    assert preds.shape == (4, 2)
    ckpt = tmp_path / "big.pt"
    model.save(ckpt)
    back = WearRoughnessDL.load(ckpt)
    assert np.allclose(back.predict(x[:4]), preds)
    freeze_backbone(back)
    trainable = [n for n, p in back.model.named_parameters() if p.requires_grad]
    assert trainable and all("head" in n for n in trainable)
