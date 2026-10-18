"""Milling-native wear regressor for LUH/KIT signal features.

Gradient boosting on the LUH featuriser's 16 signal statistics plus
``log1p`` of the run's cumulative tool-contact time. Single target:
flank wear ``V_B`` (mm) only — LUH carries no measured roughness column,
so unlike the Vicomtech turning tier this model does not predict Ra.

Mirrors the ``fit`` / ``predict`` / ``evaluate`` / ``save`` / ``load``
shape of the other models so dashboards can treat it uniformly; it
predicts from signal feature rows, not from operating points.

Honest generalisation metric (not the in-sample fit): 9-tool
leave-one-tool-out pooled RMSE 0.0068 mm against a 0.0302 mm train-mean
baseline (95% skill) over the full 6,418-run dataset, written by
``scripts/training/luh_loto.py`` to ``docs/luh_loto.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybrid_ai.core.luh_loader import FEATURE_NAMES


@dataclass
class LUHConfig:
    """HistGradientBoosting settings; matches the Vicomtech tier."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    seed: int = 42


class LUHMillingModel:
    """Single-output GBM: ``[signals (16), log1p(contact_s)] -> V_B (mm)``."""

    def __init__(self, config: LUHConfig | None = None) -> None:
        self.config = config or LUHConfig()
        self.model = None
        self.feature_names = list(FEATURE_NAMES)

    @staticmethod
    def assemble(signals: np.ndarray, contact_s: np.ndarray) -> np.ndarray:
        """Stack signal features with ``log1p`` contact time (trained layout).

        Parameters
        ----------
        signals
            ``(n, 16)`` rows in ``luh_loader.FEATURE_NAMES`` order.
        contact_s
            ``(n,)`` cumulative tool-contact time in seconds.

        Returns
        -------
        numpy.ndarray
            ``(n, 17)`` feature matrix the model consumes.
        """
        s = np.asarray(signals, dtype=np.float64)
        log_c = np.log1p(np.asarray(contact_s, dtype=np.float64)).reshape(-1, 1)
        return np.column_stack([s, log_c])

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
        self.model.fit(x, np.asarray(y, dtype=np.float64).ravel())
        return {"n_iter": int(self.model.n_iter_)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        preds = self.predict(x)
        y = np.asarray(y, dtype=np.float64).ravel()
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "n": int(y.shape[0]),
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        import joblib

        joblib.dump(
            {"model": self.model, "config": self.config,
             "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LUHMillingModel:
        import joblib

        blob = joblib.load(path)
        wrapper = cls(config=blob.get("config", LUHConfig()))
        wrapper.model = blob["model"]
        wrapper.feature_names = list(blob.get("feature_names", FEATURE_NAMES))
        return wrapper
