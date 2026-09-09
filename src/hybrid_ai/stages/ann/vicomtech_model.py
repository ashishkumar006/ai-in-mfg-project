"""Turning-native wear regressor for Vicomtech 1-second segments.

Gradient boosting on the dataset's 152 native features (it handles
NaNs internally, so no imputation leaks statistics across tools).
Mirrors the ``evaluate``/``save``/``load`` shape of the other models
so dashboards can treat it uniformly; it predicts from Vicomtech
feature rows, not from operating points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VicomtechConfig:
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    seed: int = 42


class VicomtechTurningModel:
    """Multi-output GBM: ``X (N, 152)`` -> ``[VB_mm, Ra_um]``."""

    def __init__(self, config: VicomtechConfig | None = None) -> None:
        self.config = config or VicomtechConfig()
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        self.model = MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.config.seed,
        ))
        self.model.fit(x, y)
        return {"n_iter": [e.n_iter_ for e in self.model.estimators_]}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        preds = self.predict(x)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        return {"rmse": rmse, "mae": mae, "n": int(y.shape[0])}

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        import joblib

        joblib.dump({"model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path: str | Path) -> VicomtechTurningModel:
        import joblib

        blob = joblib.load(path)
        wrapper = cls(config=blob.get("config", VicomtechConfig()))
        wrapper.model = blob["model"]
        return wrapper
