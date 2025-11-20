import pandas as pd
import numpy as np

def create_features(df):
    """
    Generates time-series features for the monthly revenue dataset.
    """
    df = df.copy()
    
    # Ensure sorted by date
    df = df.sort_values("year_month").reset_index(drop=True)
    
    # Time features
    # Convert year_month to datetime for extraction
    temp_date = pd.to_datetime(df["year_month"], format="%Y-%m")
    df["month"] = temp_date.dt.month
    df["year"] = temp_date.dt.year
    df["quarter"] = temp_date.dt.quarter
    df["month_index"] = df.index  # 0, 1, 2...
    
    # Identify numeric columns to generate features for
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["month", "year", "quarter", "month_index"]
    target_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # Lags
    for col in target_cols:
        for lag in [1, 2, 3, 6, 12]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
    # Rolling Means (shifted by 1 to avoid leakage)
    # MA_3 at time t is mean(t-1, t-2, t-3)
    for col in target_cols:
        for window in [3, 6]:
            df[f"{col}_ma_{window}"] = df[col].shift(1).rolling(window=window).mean()
        
    # Percentage Change
    for col in target_cols:
        df[f"{col}_pct_change"] = df[col].pct_change()
        df[f"{col}_pct_change_lag_1"] = df[f"{col}_pct_change"].shift(1)
    
    # Drop rows with NaNs created by lags/rolling (optional, or handle later)
    # For now, we keep them but models might need to drop them.
    
    return df
