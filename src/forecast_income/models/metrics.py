from __future__ import annotations

import numpy as np

def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def bias(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))

def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(actual)"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0)

def forecast_accuracy(y_true, y_pred) -> float:
    """1 - WAPE (Business Accuracy)"""
    return max(0.0, 100.0 - wape(y_true, y_pred))
