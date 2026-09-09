"""Remaining-useful-life models for per-cycle feature rows (Haas data).

Two options: :class:`RulModel` (gradient boosting on one cycle) and
:class:`RulSequenceModel` (GRU over a tool's cycle history — the right
shape for RUL, since remaining life depends on the trajectory, not a
single snapshot).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RulConfig:
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    seed: int = 42


class RulModel:
    """HGB regressor: ``X (N, F)`` -> cycles remaining ``(N,)``."""

    def __init__(self, config: RulConfig | None = None) -> None:
        self.config = config or RulConfig()
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.config.seed,
        )
        self.model.fit(x, y.ravel())
        return {"n_iter": int(self.model.n_iter_)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.asarray(self.model.predict(x), dtype=np.float64).ravel()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        preds = self.predict(x)
        rmse = float(np.sqrt(mean_squared_error(y.ravel(), preds)))
        mae = float(mean_absolute_error(y.ravel(), preds))
        return {"rmse": rmse, "mae": mae, "n": int(np.asarray(y).shape[0])}

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        import joblib

        joblib.dump({"model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path: str | Path) -> RulModel:
        import joblib

        blob = joblib.load(path)
        wrapper = cls(config=blob.get("config", RulConfig()))
        wrapper.model = blob["model"]
        return wrapper


@dataclass
class RulSeqConfig:
    proj_dim: int = 32
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    seed: int = 42


class RulSequenceModel:
    """GRU over a tool's cycle history -> cycles remaining per step.

    Expects padded batches ``X (B, T, F)`` with ``lengths (B,)``.
    """

    def __init__(self, config: RulSeqConfig | None = None) -> None:
        self.config = config or RulSeqConfig()
        self.model = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _build(self, in_dim: int):
        import torch
        from torch import nn

        cfg = self.config

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(in_dim, cfg.proj_dim)
                self.gru = nn.GRU(cfg.proj_dim, cfg.hidden, cfg.layers,
                                  batch_first=True,
                                  dropout=cfg.dropout if cfg.layers > 1 else 0.0)
                self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                          nn.Linear(cfg.hidden, 1))

            def forward(self, x, lengths):
                h = torch.relu(self.proj(x))
                packed = nn.utils.rnn.pack_padded_sequence(
                    h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                out, _ = self.gru(packed)
                seq, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
                return self.head(seq).squeeze(-1)

        return _Net()

    def fit(self, seqs: list[tuple[np.ndarray, np.ndarray]],
            val_seqs: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
        import torch

        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        all_x = np.concatenate([s[0] for s in seqs], axis=0)
        self.mean = all_x.mean(axis=0)
        self.std = all_x.std(axis=0) + 1e-8
        all_y = np.concatenate([s[1] for s in seqs], axis=0)
        self.y_mean = float(all_y.mean())
        self.y_std = float(all_y.std()) + 1e-8

        def _batch(items):
            t_max = max(s[0].shape[0] for s in items)
            f = items[0][0].shape[1]
            xb = np.zeros((len(items), t_max, f))
            yb = np.zeros((len(items), t_max))
            mask = np.zeros((len(items), t_max))
            ln = []
            for i, (sx, sy) in enumerate(items):
                n = sx.shape[0]
                xb[i, :n] = (sx - self.mean) / self.std
                yb[i, :n] = (sy - self.y_mean) / self.y_std
                mask[i, :n] = 1.0
                ln.append(n)
            return (torch.tensor(xb, dtype=torch.float32),
                    torch.tensor(yb, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32),
                    torch.tensor(ln, dtype=torch.long))

        self.model = self._build(all_x.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
        xb, yb, mask, ln = _batch(seqs)
        vb = None
        if val_seqs is not None:
            vb = _batch(val_seqs)
        best = float("inf")
        best_state = None
        since_best = 0
        history: list[float] = []
        for _ in range(cfg.epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(xb, ln)
            loss = (((pred - yb) ** 2) * mask).sum() / mask.sum()
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
            if vb is not None:
                self.model.eval()
                with torch.no_grad():
                    vp = self.model(vb[0], vb[3])
                    vl = float(((((vp - vb[1]) ** 2) * vb[2]).sum()
                                / vb[2].sum()).item())
                if vl < best - 1e-9:
                    best = vl
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    since_best = 0
                else:
                    since_best += 1
                if since_best >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return {"loss_history": history, "epochs_run": len(history),
                "best_val_loss": best if vb is not None else None}

    def predict_seq(self, x: np.ndarray) -> np.ndarray:
        """Predict remaining cycles for one tool's ``(T, F)`` history."""
        import torch

        if self.model is None or self.mean is None:
            raise RuntimeError("Model is not fitted yet.")
        self.model.eval()
        with torch.no_grad():
            xn = torch.tensor((x - self.mean) / self.std,
                              dtype=torch.float32).unsqueeze(0)
            out = self.model(xn, torch.tensor([x.shape[0]])).squeeze(0).numpy()
        return np.asarray(out, dtype=np.float64) * self.y_std + self.y_mean

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        import torch

        torch.save({"state_dict": self.model.state_dict(), "config": self.config,
                    "mean": self.mean, "std": self.std,
                    "y_mean": self.y_mean, "y_std": self.y_std,
                    "in_dim": int(self.mean.shape[0])}, path)

    @classmethod
    def load(cls, path: str | Path) -> RulSequenceModel:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = cls(config=blob.get("config", RulSeqConfig()))
        wrapper.model = wrapper._build(int(blob["in_dim"]))
        wrapper.model.load_state_dict(blob["state_dict"])
        wrapper.model.eval()
        wrapper.mean = blob["mean"]
        wrapper.std = blob["std"]
        wrapper.y_mean = float(blob.get("y_mean", 0.0))
        wrapper.y_std = float(blob.get("y_std", 1.0))
        return wrapper
