from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
import yaml

from forecast_income.features.feature_engineering import FeatureConfig, make_supervised_monthly
from forecast_income.utils.config import load_config

def predict_next_month(params_path: str = "config/params.yaml") -> Dict[str, Any]:
    params = load_config(params_path)
    master = pd.read_csv(params["data"]["master_table_path"], parse_dates=[params["data"]["index_col"]]).sort_values(params["data"]["index_col"]).reset_index(drop=True)

    cfg = FeatureConfig(
        base_cols=params["features"]["base_cols"],
        lags=params["features"]["lags"],
        rolling_windows=params["features"]["rolling_windows"],
        add_seasonality=params["features"]["add_seasonality"],
        add_trend=params["features"]["add_trend"],
        target_col=params["features"]["target_col"],
    )

    if params.get("active_mode") == "daily":
        from forecast_income.features.feature_engineering import create_features
        
        # Load selected features
        selected_features_path = Path(params["modeling"]["export_path"]).parent / "selected_features.json"
        selected_cols = None
        if selected_features_path.exists():
            selected_cols = json.loads(selected_features_path.read_text(encoding="utf-8"))
            
        model = joblib.load(params["modeling"]["export_path"])
        
        # Recursive prediction for next 30 days
        days_to_predict = 30
        current_df = master.copy()
        future_revenue = 0
        
        # Ensure date col is datetime
        date_col = params["data"]["index_col"]
        current_df[date_col] = pd.to_datetime(current_df[date_col])
        
        for _ in range(days_to_predict):
            # Create features on tail
            tail_len = 60
            df_tail = current_df.iloc[-tail_len:].copy()
            
            df_feats, _ = create_features(df_tail, cfg, date_col=date_col)
            
            # Predict using last row features
            X_pred = df_feats.iloc[[-1]]
            
            if selected_cols:
                X_pred = X_pred[selected_cols]
                
            pred = float(model.predict(X_pred)[0])
            pred = max(0.0, pred) # Clip negative predictions
            future_revenue += pred
            
            # Append prediction
            last_date = current_df.iloc[-1][date_col]
            next_date = last_date + pd.Timedelta(days=1)
            
            new_row = current_df.iloc[[-1]].copy()
            new_row[date_col] = next_date
            new_row["daily_revenue"] = pred
            # Exogenous variables are repeated (naive)
            
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            
        return {
            "last_observed_date": master[date_col].iloc[-1].isoformat(),
            "predicted_days": days_to_predict,
            "prediction_next_30_days_revenue": future_revenue
        }

    else:
        X, y, feats, _ = make_supervised_monthly(master, cfg)

        # Load selected features
        selected_features_path = Path(params["modeling"]["export_path"]).parent / "selected_features.json"
        if selected_features_path.exists():
            selected_cols = json.loads(selected_features_path.read_text(encoding="utf-8"))
            X = X[selected_cols]
        
        # Para predecir el próximo mes, usamos el último vector de features disponible (el último mes con lags completos).
        X_last = X.iloc[[-1]]
        model = joblib.load(params["modeling"]["export_path"])
        y_hat = float(model.predict(X_last)[0])

        last_month = master.sort_values("month")["month"].iloc[-1]
        next_month = (last_month + pd.offsets.MonthBegin(1)).to_pydatetime().date().isoformat()

        return {
            "last_observed_month": last_month.date().isoformat(),
            "predicted_month": next_month,
            "prediction_monthly_revenue": y_hat,
        }
