from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from forecast_income.utils.time_series import add_month_index, add_month_sin_cos

@dataclass(frozen=True)
class FeatureConfig:
    base_cols: List[str]
    lags: List[int]
    rolling_windows: List[int]
    add_seasonality: bool = True
    add_trend: bool = True
    target_col: str = "monthly_revenue_next"

def create_features(df: pd.DataFrame, cfg: FeatureConfig, date_col: str = "month") -> Tuple[pd.DataFrame, List[str]]:
    """
    Genera un set extendido de features para el modelo de regresión.
    Usa la configuración provista en cfg.
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)
    
    generated_feats = []
    
    # 1. Transformaciones por columna base
    new_features = []
    for col in cfg.base_cols:
        if col not in df.columns:
            continue
            
        # Lags extendidos
        for lag in cfg.lags:
            feat_name = f"{col}_lag_{lag}"
            new_features.append(df[col].shift(lag).rename(feat_name))
            generated_feats.append(feat_name)
            
        # Rolling Stats extendidos
        for window in cfg.rolling_windows:
            # Shift(1) para evitar data leakage (usar info hasta t-1)
            shifted = df[col].shift(1)
            
            # Mean
            feat_mean = f"{col}_roll_mean_{window}"
            new_features.append(shifted.rolling(window=window, min_periods=1).mean().rename(feat_mean))
            generated_feats.append(feat_mean)
            
            # Std
            feat_std = f"{col}_roll_std_{window}"
            new_features.append(shifted.rolling(window=window, min_periods=1).std().rename(feat_std))
            generated_feats.append(feat_std)
            
            # Min
            feat_min = f"{col}_roll_min_{window}"
            new_features.append(shifted.rolling(window=window, min_periods=1).min().rename(feat_min))
            generated_feats.append(feat_min)
            
            # Max
            feat_max = f"{col}_roll_max_{window}"
            new_features.append(shifted.rolling(window=window, min_periods=1).max().rename(feat_max))
            generated_feats.append(feat_max)
            
        # Delta (pct_change) - Lag 1
        feat_delta = f"{col}_pct_change"
        pct_change = df[col].pct_change().shift(1)
        # Replace inf with 0 and fill NaNs with 0
        pct_change = pct_change.replace([np.inf, -np.inf], 0).fillna(0)
        new_features.append(pct_change.rename(feat_delta))
        generated_feats.append(feat_delta)
    
    # Concatenar todas las nuevas features de una vez
    if new_features:
        df = pd.concat([df] + new_features, axis=1)
            
    # 2. Features de tiempo    # 2. Features de tiempo
    time_feats = []
    
    # Usamos date_col para extraer features temporales
    if date_col in df.columns:
        # Asegurar que es datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = pd.to_datetime(df[date_col])

        quarter_feat = df[date_col].dt.quarter.rename("quarter")
        time_feats.append(quarter_feat)
        generated_feats.append("quarter")
        
        month_num_feat = df[date_col].dt.month.rename("month_num")
        time_feats.append(month_num_feat)
        generated_feats.append("month_num")
        
        # Para daily, podemos agregar day_of_week y day_of_month
        if date_col == "date":
            dow_feat = df[date_col].dt.dayofweek.rename("day_of_week")
            time_feats.append(dow_feat)
            generated_feats.append("day_of_week")
            
            dom_feat = df[date_col].dt.day.rename("day_of_month")
            time_feats.append(dom_feat)
            generated_feats.append("day_of_month")

    if time_feats:
        df = pd.concat([df] + time_feats, axis=1)
    
    # 3. Cíclicas
    # Verificamos duplicados y nos quedamos con uno
    df = df.loc[:, ~df.columns.duplicated()]
    
    if "month_num" in df.columns:
        df = add_month_sin_cos(df, month_col="month_num")
        generated_feats.append("month_sin")
        generated_feats.append("month_cos")
    
    return df, generated_feats

def make_supervised_monthly(master: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
    """
    Transforma master table mensual a dataset supervisado (horizonte 1 mes)
    usando create_features.
    """
    # Generar features
    df, feats = create_features(master, cfg, date_col="month")
    
    # Target 1 mes adelante
    df[cfg.target_col] = df["monthly_revenue"].shift(-1)
    
    supervised = df.dropna(subset=feats + [cfg.target_col]).reset_index(drop=True)
    
    X = supervised[feats]
    y = supervised[cfg.target_col]
    dates = supervised["month"]
    
    print(f"Total features generated: {len(feats)}")
    
    return X, y, feats, dates

def make_supervised_daily(master: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
    """
    Transforma master table diaria a dataset supervisado (horizonte 1 día)
    usando create_features.
    """
    # Generar features
    df, feats = create_features(master, cfg, date_col="date")
    
    # Target 1 día adelante
    df[cfg.target_col] = df["daily_revenue"].shift(-1)
    
    supervised = df.dropna(subset=feats + [cfg.target_col]).reset_index(drop=True)
    
    X = supervised[feats]
    y = supervised[cfg.target_col]
    dates = supervised["date"]
    
    print(f"Total features generated: {len(feats)}")
    
    return X, y, feats, dates

