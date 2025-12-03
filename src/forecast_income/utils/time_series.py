import numpy as np
import pandas as pd

def add_month_sin_cos(df: pd.DataFrame, month_col: str = "month_num") -> pd.DataFrame:
    """Codifica cíclica del mes (1..12) -> sin/cos."""
    out = df.copy()
    m = out[month_col].astype(int)
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    return out

def add_month_index(df: pd.DataFrame, date_col: str = "month") -> pd.DataFrame:
    """Tendencia simple como índice temporal 0..n-1."""
    out = df.copy()
    out = out.sort_values(date_col)
    out["month_index"] = np.arange(len(out))
    return out
