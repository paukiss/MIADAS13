from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class NaiveLastValue:
    """Predice y(t+1) = y(t)."""
    last_value_: float | None = None

    def fit(self, y: pd.Series):
        self.last_value_ = float(y.iloc[-1])
        return self

    def predict(self, y_hist: pd.Series, steps: int = 1) -> np.ndarray:
        if len(y_hist) == 0:
            raise ValueError("y_hist vacío")
        return np.array([float(y_hist.iloc[-1])] * steps)

@dataclass
class SeasonalNaive:
    season_lag: int = 12

    def fit(self, y: pd.Series):
        return self

    def predict(self, y_hist: pd.Series, steps: int = 1) -> np.ndarray:
        preds = []
        for i in range(steps):
            idx = -self.season_lag + i
            if abs(idx) > len(y_hist):
                preds.append(float(y_hist.iloc[-1]))
            else:
                preds.append(float(y_hist.iloc[idx]))
        return np.array(preds)
