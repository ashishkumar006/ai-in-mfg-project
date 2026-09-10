"""ANN wear / roughness predictor.

A small PyTorch MLP that takes the same feature vector as
:func:`hybrid_ai.core.synthetic.generate_synthetic_dataset` and
predicts ``[V_B, R_a]``. A scikit-learn ``GradientBoostingRegressor`` is
also provided as a baseline for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from hybrid_ai.core.schemas import (
    OperatingPoint,
    SensorReading,
    ToolCandidate,
    WearPrediction,
)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden: tuple[int, int] = (32, 32)
    seed: int = 42


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WearRoughnessModel:
    """Wrap an MLP plus its input normaliser."""

    def __init__(self) -> None:
        self.model: _MLP | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: TrainConfig | None = None,
    ) -> dict:
        config = config or TrainConfig()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        self.target_mean = y.mean(axis=0)
        self.target_std = y.std(axis=0) + 1e-8

        x_n = (x - self.mean) / self.std
        y_n = (y - self.target_mean) / self.target_std

        x_t = torch.tensor(x_n, dtype=torch.float32)
        y_t = torch.tensor(y_n, dtype=torch.float32)

        self.model = _MLP(x.shape[1], y.shape[1], config.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        loss_fn = nn.MSELoss()

        history: list[float] = []
        n = x_t.shape[0]
        for _ in range(config.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, config.batch_size):
                idx = perm[i : i + config.batch_size]
                pred = self.model(x_t[idx])
                loss = loss_fn(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * idx.numel()
            history.append(epoch_loss / n)
        return {"loss_history": history, "final_loss": history[-1]}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("Model is not fitted yet.")
        x_n = (x - self.mean) / self.std
        with torch.no_grad():
            y_n = self.model(torch.tensor(x_n, dtype=torch.float32)).numpy()
        return y_n * self.target_std + self.target_mean

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(x)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        return {"rmse": rmse, "mae": mae, "n": int(y.shape[0])}

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden": tuple(
                    int(m.out_features)
                    for m in self.model.net
                    if isinstance(m, nn.Linear)
                )[:-1],
                "in_dim": int(self.model.net[0].in_features),
                "out_dim": int(self.model.net[-1].out_features),
                "mean": self.mean,
                "std": self.std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        in_dim = int(blob["in_dim"])
        out_dim = int(blob["out_dim"])
        hidden = tuple(blob["hidden"])
        self.model = _MLP(in_dim, out_dim, hidden)
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.mean = blob["mean"]
        self.std = blob["std"]
        self.target_mean = blob["target_mean"]
        self.target_std = blob["target_std"]


def predict_for_point(
    model: WearRoughnessModel,
    tool: ToolCandidate,
    point: OperatingPoint,
    hardness_hb: float = 200.0,
    sensor: SensorReading | None = None,
    n_mc: int = 0,
) -> WearPrediction:
    """Predict V_B and R_a for a concrete tool + operating point.

    Builds the 11-element feature vector ``[V_c, f, ap, nose_radius,
    rake_angle, iso_P..iso_S, hardness]`` and clips the output to
    physical ranges. If the fitted model expects the 51-element
    NASA-style vector (4 scalar + 5 ISO + 42 sensor feats) and a
    ``sensor`` reading is given, the sensor block falls back to the
    model's training mean so the query stays in-distribution; the
    prediction still varies with V_c / feed / ap / hardness / ISO.

    With ``n_mc > 0`` on a deep window model, the mean/std come from
    Monte-Carlo dropout passes and ``uncertainty`` reports the V_B
    spread in mm (0.0 when not estimated).
    """
    model_mean = getattr(model, "mean", None)
    if sensor is not None and model_mean is not None and model_mean.shape[-1] == 51:
        iso_codes = ["P", "M", "K", "N", "S"]
        iso_onehot = [1.0 if code == tool.iso_group else 0.0 for code in iso_codes]
        sensor_base = np.asarray(model_mean.ravel()[9:], dtype=np.float64)
        x = np.array(
            [[point.v_c_mpm, point.f_mmpr, point.ap_mm, hardness_hb, *iso_onehot, *sensor_base]],
            dtype=np.float64,
        )
    else:
        iso_codes = ["P", "M", "K", "N", "S"]
        iso_onehot = [1.0 if code == tool.iso_group else 0.0 for code in iso_codes]
        x = np.array(
            [
                [
                    point.v_c_mpm,
                    point.f_mmpr,
                    point.ap_mm,
                    tool.nose_radius_mm,
                    tool.rake_angle_deg,
                    *iso_onehot,
                    hardness_hb,
                ]
            ],
            dtype=np.float64,
        )
    pred = model.predict(x)[0]
    uncertainty = 0.0
    if n_mc > 0 and isinstance(model, WearRoughnessDL):
        mean, std = model.predict_with_uncertainty(x, n_passes=n_mc)
        pred = mean[0]
        uncertainty = float(std[0][0])
    v_b = float(np.clip(pred[0], 0.02, 1.5))
    ra = float(np.clip(pred[1], 0.2, 6.0))
    confidence = float(
        max(0.0, min(1.0, 1.0 - abs(v_b - 0.25) / 1.0))
    )
    return WearPrediction(
        tool=tool,
        point=point,
        v_b_mm=v_b,
        ra_um=ra,
        model_confidence=confidence,
        uncertainty=uncertainty,
    )


class SklearnBaseline:
    """scikit-learn baseline (two GBRT heads) used for benchmarking."""

    def __init__(self) -> None:
        self.v_b = GradientBoostingRegressor(n_estimators=120, random_state=42)
        self.ra = GradientBoostingRegressor(n_estimators=120, random_state=42)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        x_n = (x - self.mean) / self.std
        self.v_b.fit(x_n, y[:, 0])
        self.ra.fit(x_n, y[:, 1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_n = (x - self.mean) / self.std
        v_b = self.v_b.predict(x_n)
        ra = self.ra.predict(x_n)
        return np.stack([v_b, ra], axis=1)


# ---------------------------------------------------------------------------
# 1D-CNN and CNN-LSTM models for raw sensor windows
# ---------------------------------------------------------------------------


class _CNN1D(nn.Module):
    """1D-CNN for fixed-length sensor windows.

    Input: (batch, n_channels, window_len)
    Output: (batch, 2) -> [VB, Ra]
    """

    def __init__(
        self,
        n_channels: int,
        window_len: int,
        out_dim: int = 2,
        *,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, window_len)
            flat = self.features(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _CNNLSTM(nn.Module):
    """CNN front-end + LSTM + attention pooling.

    Input: (batch, n_channels, window_len)
    Output: (batch, 2) -> [VB, Ra]
    """

    def __init__(
        self,
        n_channels: int,
        window_len: int,
        out_dim: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(lstm_hidden, 1)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = f.transpose(1, 2)
        out, _ = self.lstm(f)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.head(ctx)


class _CNNBiLSTM(nn.Module):
    """Large CNN + BiLSTM + attention model for PHM-scale window data.

    Three conv blocks (with GroupNorm) downsample the signal 8x, a
    2-layer BiLSTM reads the sequence both ways, attention pooling
    picks the informative steps, and a linear head maps to [VB, Ra].
    Roughly 1M parameters at the default width (vs ~74k for _CNNLSTM).

    Input: (batch, n_channels, window_len)
    Output: (batch, 2) -> [VB, Ra]
    """

    def __init__(
        self,
        n_channels: int,
        window_len: int,
        out_dim: int = 2,
        conv_channels: tuple[int, int, int] = (64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels

        def _groups(ch: int, want: int = 8) -> int:
            import math

            g = math.gcd(ch, want)
            return g if g > 1 else 1

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, c1, kernel_size=7, padding=3),
            nn.GroupNorm(_groups(c1), c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(c2), c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.GroupNorm(_groups(c3, 16), c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = f.transpose(1, 2)
        out, _ = self.lstm(f)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.head(ctx)


@dataclass
class DLConfig:
    """Hyperparameters for CNN1D, CNN-LSTM and CNN-BiLSTM."""

    model_type: str = "cnn1d"  # "cnn1d" | "cnn_lstm" | "cnn_bilstm"
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    n_channels: int = 6
    window_len: int = 9000
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.2
    conv_channels: tuple[int, int, int] = (64, 128, 256)
    weight_decay: float = 0.0
    patience: int | None = None
    min_delta: float = 0.0
    verbose: bool = False


class WearRoughnessDL:
    """Wrapper that mirrors ``WearRoughnessModel`` but supports the deep
    window architectures (CNN1D / CNN-LSTM / CNN-BiLSTM).

    The public API is identical to ``WearRoughnessModel`` so downstream
    modules (Fuzzy, NSGA-II, dashboard) do not need to change.
    """

    def __init__(self, config: DLConfig | None = None) -> None:
        self.config = config or DLConfig()
        self.model: nn.Module | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None
        self.input_shape: tuple[int, ...] | None = None

    def _build(self, x: np.ndarray) -> nn.Module:
        cfg = self.config
        if cfg.model_type == "cnn1d":
            return _CNN1D(
                n_channels=cfg.n_channels,
                window_len=cfg.window_len,
                dropout=cfg.dropout,
            )
        if cfg.model_type == "cnn_lstm":
            return _CNNLSTM(
                n_channels=cfg.n_channels,
                window_len=cfg.window_len,
                lstm_hidden=cfg.lstm_hidden,
                lstm_layers=cfg.lstm_layers,
                dropout=cfg.dropout,
            )
        if cfg.model_type == "cnn_bilstm":
            return _CNNBiLSTM(
                n_channels=cfg.n_channels,
                window_len=cfg.window_len,
                conv_channels=tuple(cfg.conv_channels),
                lstm_hidden=cfg.lstm_hidden,
                lstm_layers=cfg.lstm_layers,
                dropout=cfg.dropout,
            )
        raise ValueError(f"Unknown model_type {cfg.model_type!r}")

    def fit(self, x: np.ndarray, y: np.ndarray, config: DLConfig | None = None,
            val_data: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
        config = config or self.config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if x.ndim == 2:
            x = x[:, np.newaxis, :]

        self.mean = x.mean(axis=(0, 2), keepdims=True)
        self.std = x.std(axis=(0, 2), keepdims=True) + 1e-8
        self.target_mean = y.mean(axis=0)
        self.target_std = y.std(axis=0) + 1e-8

        x_n = (x - self.mean) / self.std
        y_n = (y - self.target_mean) / self.target_std

        x_t = torch.tensor(x_n, dtype=torch.float32)
        y_t = torch.tensor(y_n, dtype=torch.float32)

        xv_t = yv_t = None
        if val_data is not None:
            xv, yv = val_data
            if xv.ndim == 2:
                xv = xv[:, np.newaxis, :]
            xv_t = torch.tensor((xv - self.mean) / self.std, dtype=torch.float32)
            yv_t = torch.tensor((yv - self.target_mean) / self.target_std,
                                dtype=torch.float32)

        self.input_shape = (x.shape[1], x.shape[2])
        self.model = self._build(x)
        opt = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
        loss_fn = nn.MSELoss()

        history: list[float] = []
        best_val = float("inf")
        best_state: dict | None = None
        since_best = 0
        stopped_early = False
        n = x_t.shape[0]
        for epoch in range(config.epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, config.batch_size):
                idx = perm[i : i + config.batch_size]
                pred = self.model(x_t[idx])
                loss = loss_fn(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * idx.numel()
            history.append(epoch_loss / n)
            if xv_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(self.model(xv_t), yv_t).item())
                if config.verbose:
                    print(f"epoch {epoch + 1}/{config.epochs} "
                          f"train={history[-1]:.4f} val={val_loss:.4f}",
                          flush=True)
                if val_loss < best_val - config.min_delta:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    since_best = 0
                else:
                    since_best += 1
                if config.patience is not None and since_best >= config.patience:
                    stopped_early = True
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.eval()
        out = {"loss_history": history, "final_loss": history[-1],
               "epochs_run": len(history), "stopped_early": stopped_early}
        if xv_t is not None:
            out["best_val_loss"] = best_val
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("Model is not fitted yet.")
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        self.model.eval()
        x_n = (x - self.mean) / self.std
        with torch.no_grad():
            y_n = self.model(torch.tensor(x_n, dtype=torch.float32)).numpy()
        return y_n * self.target_std + self.target_mean

    def predict_with_uncertainty(
        self, x: np.ndarray, n_passes: int = 20, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo dropout: mean + std over stochastic forward passes.

        Dropout layers stay on (GroupNorm is batch-independent, so the
        statistics stay valid); models without dropout return ~zero
        spread, honestly reporting "no uncertainty source".
        """
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("Model is not fitted yet.")
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        torch.manual_seed(seed)
        x_n = (x - self.mean) / self.std
        x_t = torch.tensor(x_n, dtype=torch.float32)
        self.model.train()
        try:
            with torch.no_grad():
                draws = np.stack(
                    [self.model(x_t).numpy() for _ in range(max(1, n_passes))],
                    axis=0,
                )
        finally:
            self.model.eval()
        draws = draws * self.target_std + self.target_mean
        return draws.mean(axis=0), draws.std(axis=0)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(x)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        return {"rmse": rmse, "mae": mae, "n": int(y.shape[0])}

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_type": self.config.model_type,
                "n_channels": self.config.n_channels,
                "window_len": self.config.window_len,
                "input_shape": self.input_shape,
                "mean": self.mean,
                "std": self.std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "conv_channels": list(self.config.conv_channels),
                "lstm_hidden": self.config.lstm_hidden,
                "lstm_layers": self.config.lstm_layers,
                "dropout": self.config.dropout,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, config: DLConfig | None = None) -> WearRoughnessDL:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model_type = str(blob.get("model_type", "cnn1d"))
        cfg = config or DLConfig(
            model_type=model_type,
            n_channels=int(blob.get("n_channels", 6)),
            window_len=int(blob.get("window_len", 9000)),
            conv_channels=tuple(int(c) for c in blob.get("conv_channels", (64, 128, 256))),
            lstm_hidden=int(blob.get("lstm_hidden", 64)),
            lstm_layers=int(blob.get("lstm_layers", 2)),
            dropout=float(blob.get("dropout", 0.2)),
        )
        wrapper = cls(config=cfg)
        wrapper.input_shape = tuple(blob.get("input_shape", (cfg.n_channels, cfg.window_len)))
        wrapper.model = wrapper._build(np.zeros((1, wrapper.input_shape[0], wrapper.input_shape[1])))
        wrapper.model.load_state_dict(blob["state_dict"])
        wrapper.model.eval()
        wrapper.mean = blob["mean"]
        wrapper.std = blob["std"]
        wrapper.target_mean = blob["target_mean"]
        wrapper.target_std = blob["target_std"]
        return wrapper


# ---------------------------------------------------------------------------
# Transfer learning helper
# ---------------------------------------------------------------------------


def freeze_backbone(model: WearRoughnessDL) -> WearRoughnessDL:
    """Freeze everything except the final regression head.

    The "backbone" is the CNN feature extractor + (optional) LSTM.
    Only the final ``Linear`` layer that maps to ``[VB, Ra]`` stays
    trainable. Returns the same model object for chaining.
    """
    if model.model is None:
        raise RuntimeError("Model is not fitted yet.")
    if model.config.model_type == "cnn1d":
        for p in model.model.features.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True
    elif model.config.model_type == "cnn_lstm":
        for p in model.model.features.parameters():
            p.requires_grad = False
        for p in model.model.lstm.parameters():
            p.requires_grad = False
        for p in model.model.attn.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True
    elif model.config.model_type == "cnn_bilstm":
        for p in model.model.features.parameters():
            p.requires_grad = False
        for p in model.model.lstm.parameters():
            p.requires_grad = False
        for p in model.model.attn.parameters():
            p.requires_grad = False
        for p in model.model.head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model_type {model.config.model_type!r}")
    return model


def unfreeze_all(model: WearRoughnessDL) -> WearRoughnessDL:
    """Unfreeze every parameter in the model."""
    if model.model is None:
        raise RuntimeError("Model is not fitted yet.")
    for p in model.model.parameters():
        p.requires_grad = True
    return model


def fine_tune_head(
    pretrained_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    unfreeze_after: int | None = None,
    batch_size: int = 32,
    seed: int = 42,
) -> WearRoughnessDL:
    """Load a pretrained CNN/CNN-LSTM, freeze the backbone, replace
    the head, and fine-tune on a small target dataset.

    Parameters
    ----------
    pretrained_path:
        Path to a ``.pt`` file produced by :meth:`WearRoughnessDL.save`.
    x, y:
        Target dataset (e.g. NASA windows). ``x`` may be 2D or 3D.
    epochs:
        Number of fine-tuning epochs.
    learning_rate:
        Optimiser learning rate.
    unfreeze_after:
        If given, unfreeze the entire model after this many epochs
        and continue training at the same learning rate.
    batch_size:
        Mini-batch size.
    seed:
        Random seed.

    Returns
    -------
    WearRoughnessDL
        Fine-tuned model ready for ``predict`` and ``save``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = WearRoughnessDL.load(pretrained_path)
    freeze_backbone(model)

    if x.ndim == 2:
        x = x[:, np.newaxis, :]

    model.mean = x.mean(axis=(0, 2), keepdims=True)
    model.std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    model.target_mean = y.mean(axis=0)
    model.target_std = y.std(axis=0) + 1e-8

    x_n = (x - model.mean) / model.std
    y_n = (y - model.target_mean) / model.target_std
    x_t = torch.tensor(x_n, dtype=torch.float32)
    y_t = torch.tensor(y_n, dtype=torch.float32)

    opt = torch.optim.Adam(
        [p for p in model.model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    loss_fn = nn.MSELoss()

    n = x_t.shape[0]
    history: list[float] = []
    for epoch in range(epochs):
        if unfreeze_after is not None and epoch == unfreeze_after:
            unfreeze_all(model)
            opt = torch.optim.Adam(model.model.parameters(), lr=learning_rate * 0.1)
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = model.model(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * idx.numel()
        history.append(epoch_loss / n)

    return model
