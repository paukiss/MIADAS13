import pandas as pd
import numpy as np

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def wape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100.0)

df = pd.read_csv("reports/metrics/backtest_predictions.csv")
print(f"Rows: {len(df)}")
print(f"Sum y_true: {df['y_true'].sum()}")
print(f"Sum abs error: {(df['y_true'] - df['y_pred']).abs().sum()}")

m = mape(df['y_true'], df['y_pred'])
w = wape(df['y_true'], df['y_pred'])

print(f"Calculated MAPE: {m}")
print(f"Calculated WAPE: {w}")
