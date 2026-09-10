from __future__ import annotations

import numpy as np

from hybrid_ai.stages.ann.rul_model import RulModel, RulSeqConfig, RulSequenceModel
from hybrid_ai.core.haas_loader import load_haas, load_haas_sequences


def test_rul_model_trains_and_roundtrips(tmp_path) -> None:
    x, y, _ = load_haas(split="train")
    model = RulModel()
    model.fit(x[:200], y[:200])
    assert model.predict(x[:4]).shape == (4,)
    ckpt = tmp_path / "rul.joblib"
    model.save(ckpt)
    back = RulModel.load(ckpt)
    assert np.allclose(back.predict(x[:4]), model.predict(x[:4]))
    assert back.evaluate(x[:50], y[:50])["n"] == 50


def test_sequence_single_row_history() -> None:
    seqs = load_haas_sequences(split="test")
    shortest = min(seqs, key=lambda s: s[0].shape[0])
    model = RulSequenceModel(config=RulSeqConfig(epochs=3, patience=1000))
    model.fit([(shortest[0][:1], shortest[1][:1])])
    pred = model.predict_seq(shortest[0][:1])
    assert pred.shape == (1,)
    assert np.all(np.isfinite(pred))


def test_sequence_model_learns_and_roundtrips(tmp_path) -> None:
    seqs = [(x, y) for x, y, _ in load_haas_sequences(split="test")]
    model = RulSequenceModel(config=RulSeqConfig(epochs=25, patience=1000))
    history = model.fit(seqs)
    assert history["epochs_run"] == 25
    x0, y0, _ = load_haas_sequences(split="test")[0]
    pred = model.predict_seq(x0)
    assert pred.shape == y0.shape
    base = float(np.sqrt(np.mean((y0 - y0.mean()) ** 2)))
    got = float(np.sqrt(np.mean((y0 - pred) ** 2)))
    assert got < base  # fits its own training tools better than the mean
    ckpt = tmp_path / "rul_seq.pt"
    model.save(ckpt)
    back = RulSequenceModel.load(ckpt)
    assert np.allclose(back.predict_seq(x0), pred)
