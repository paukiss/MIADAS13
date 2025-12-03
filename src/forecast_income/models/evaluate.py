from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.base import clone

from forecast_income.features.feature_engineering import FeatureConfig, make_supervised_monthly, make_supervised_daily
from forecast_income.models.metrics import mae, rmse, mape, bias, wape, forecast_accuracy
from forecast_income.utils.logger import get_logger

from forecast_income.utils.config import load_config

LOGGER = get_logger(__name__)

def _load_master(path: str) -> pd.DataFrame:
    if "daily" in path:
        return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return pd.read_csv(path, parse_dates=["month"]).sort_values("month").reset_index(drop=True)

def _load_model(path: str):
    return joblib.load(path)

def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "Accuracy": forecast_accuracy(y_true, y_pred),
    }

def _backtest_expanding(model, X: pd.DataFrame, y: pd.Series, dates: pd.Series, start: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Expanding window: entrena en [0..t) y predice t, repite."""
    preds = []
    actuals = []
    months = []
    date_vals = []
    
    for t in range(start, len(X)):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test, y_test = X.iloc[t:t+1], y.iloc[t:t+1]
        m = clone(model)  # copia corta (evita leakage)
        m.fit(X_train, y_train)
        p = float(m.predict(X_test)[0])
        preds.append(p)
        actuals.append(float(y_test.iloc[0]))
        months.append(t)
        date_vals.append(dates.iloc[t])

    bt = pd.DataFrame({
        "t": months,
        "date": date_vals,
        "y_true": actuals,
        "y_pred": preds,
    })
    return bt, _compute_metrics(bt["y_true"], bt["y_pred"])

def _plot_performance(bt: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    plt.plot(bt["y_true"].values, label="Real")
    plt.plot(bt["y_pred"].values, label="Predicho")
    plt.title("Backtest (expanding window): Real vs Predicho")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Ingresos")
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _plot_feature_importance(model, feature_names: List[str], out_path: str) -> None:
    import numpy as np

    importance = None
    name = model.__class__.__name__

    # Pipeline (ridge)
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            importance = np.abs(inner.coef_)
            name = inner.__class__.__name__

    if importance is None and hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    if importance is None:
        LOGGER.info("No hay feature importance disponible para este modelo.")
        return

    imp = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(20)

    plt.figure()
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(f"Top 20 Feature Importance ({name})")
    plt.xlabel("Importancia (abs coef o impurity)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def run_backtest_and_reports(params_path: str = "config/params.yaml") -> Dict[str, Any]:
    params = load_config(params_path)

    master = _load_master(params["data"]["master_table_path"])
    cfg = FeatureConfig(
        base_cols=params["features"]["base_cols"],
        lags=params["features"]["lags"],
        rolling_windows=params["features"]["rolling_windows"],
        add_seasonality=params["features"]["add_seasonality"],
        add_trend=params["features"]["add_trend"],
        target_col=params["features"]["target_col"],
    )

    is_daily = params.get("active_mode") == "daily"

    if is_daily:
        X, y, feature_names, dates = make_supervised_daily(master, cfg)
        # Default test days for daily: 180
        test_len = params["modeling"].get("test_size", 180)
        start = max(30, len(X) - test_len)
    else:
        X, y, feature_names, dates = make_supervised_monthly(master, cfg)
        # Default test months for monthly: 6
        test_len = params["modeling"].get("test_size", 6)
        start = max(6, len(X) - test_len)

    # Load selected features
    selected_features_path = Path(params["modeling"]["export_path"]).parent / "selected_features.json"
    if selected_features_path.exists():
        selected_cols = json.loads(selected_features_path.read_text(encoding="utf-8"))
        X = X[selected_cols]
        feature_names = selected_cols
    else:
        LOGGER.warning("No se encontró selected_features.json, usando todas las features.")

    model = _load_model(params["modeling"]["export_path"])
    
    # Run backtest
    bt, metrics = _backtest_expanding(model, X, y, dates, start=start)

    # --- Business KPIs Calculation ---
    # Merge with master to get context
    cols_to_merge = ["total_orders", "total_freight", "top_category", "new_sellers", "unique_customers"]
    available_cols = [c for c in cols_to_merge if c in master.columns]
    
    if not is_daily:
        # Merge on date/month
        bt = bt.merge(master[["month"] + available_cols], left_on="date", right_on="month", how="left")
        
        # Calculate KPIs
        # 1. Growth MoM (on Real Revenue)
        bt["revenue_growth_mom"] = bt["y_true"].pct_change()
        
        # 2. Avg Ticket (AOV)
        if "total_orders" in bt.columns:
            bt["avg_ticket"] = bt["y_true"] / bt["total_orders"]

        # 3. ARPU (Average Revenue Per User)
        if "unique_customers" in bt.columns:
            bt["arpu"] = bt["y_true"] / bt["unique_customers"]
            
        # 4. Freight Ratio (Logistics Impact)
        if "total_freight" in bt.columns:
            bt["freight_ratio"] = bt["total_freight"] / bt["y_true"]

        # 5. Frequency (Orders per Customer)
        if "total_orders" in bt.columns and "unique_customers" in bt.columns:
            bt["frequency"] = bt["total_orders"] / bt["unique_customers"]
            
        # 6. Avg Freight
        if "total_freight" in bt.columns and "total_orders" in bt.columns:
            bt["avg_freight"] = bt["total_freight"] / bt["total_orders"]
            
        # Summary for the report (Average over the backtest period)
        total_real = float(bt["y_true"].sum())
        total_pred = float(bt["y_pred"].sum())
        
        business_summary = {
            "Total Revenue (Real)": total_real,
            "Total Forecast (Pred)": total_pred,
            "Forecast Realization Rate (%)": (total_pred / total_real * 100.0) if total_real > 0 else 0.0,
            "Revenue Deviation (R$)": total_pred - total_real,
            "Avg Growth MoM (%)": float(bt["revenue_growth_mom"].mean() * 100) if "revenue_growth_mom" in bt.columns else None,
            "Avg Ticket (AOV)": float(bt["avg_ticket"].mean()) if "avg_ticket" in bt.columns else None,
            "Avg ARPU": float(bt["arpu"].mean()) if "arpu" in bt.columns else None,
            "Avg Freight Ratio (%)": float(bt["freight_ratio"].mean() * 100) if "freight_ratio" in bt.columns else None,
            "Avg Orders/Customer": float(bt["frequency"].mean()) if "frequency" in bt.columns else None,
            "Total New Sellers": int(bt["new_sellers"].sum()) if "new_sellers" in bt.columns else None,
            "Top Category (Most Freq)": bt["top_category"].mode()[0] if "top_category" in bt.columns and not bt["top_category"].mode().empty else None
        }
        
        # Add to metrics
        metrics.update(business_summary)

    # If daily, aggregate to monthly for reporting
    if is_daily:
        bt["month"] = bt["date"].dt.to_period("M")
        monthly_bt = bt.groupby("month")[["y_true", "y_pred"]].sum().reset_index()
        monthly_bt["month"] = monthly_bt["month"].dt.to_timestamp()
        # Recompute metrics on monthly aggregates
        metrics = _compute_metrics(monthly_bt["y_true"], monthly_bt["y_pred"])
        # Use monthly_bt for plotting
        plot_bt = monthly_bt
    else:
        plot_bt = bt

    # save artifacts
    bt_path = Path("reports/metrics/backtest_predictions.csv")
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(bt_path, index=False)

    metrics_path = Path("reports/metrics/backtest_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_performance(plot_bt, params["modeling"]["performance_fig"])
    _plot_feature_importance(model, feature_names, params["modeling"]["feature_importance_fig"])

    return {
        "backtest_predictions_csv": str(bt_path),
        "backtest_metrics_json": str(metrics_path),
        "performance_fig": params["modeling"]["performance_fig"],
        "feature_importance_fig": params["modeling"]["feature_importance_fig"],
        "metrics": metrics,
    }

    master = _load_master(params["data"]["master_table_path"])
    cfg = FeatureConfig(
        base_cols=params["features"]["base_cols"],
        lags=params["features"]["lags"],
        rolling_windows=params["features"]["rolling_windows"],
        add_seasonality=params["features"]["add_seasonality"],
        add_trend=params["features"]["add_trend"],
        target_col=params["features"]["target_col"],
    )

    if "daily" in params["data"]["master_table_path"]:
        X, y, feature_names, dates = make_supervised_daily(master, cfg)
        start = max(30, len(X) - params["modeling"].get("test_days", 180))
    else:
        X, y, feature_names, dates = make_supervised_monthly(master, cfg)
        start = max(6, len(X) - params["modeling"].get("test_months", 6))

    # Load selected features
    selected_features_path = Path(params["modeling"]["export_path"]).parent / "selected_features.json"
    if selected_features_path.exists():
        selected_cols = json.loads(selected_features_path.read_text(encoding="utf-8"))
        X = X[selected_cols]
        feature_names = selected_cols
    else:
        LOGGER.warning("No se encontró selected_features.json, usando todas las features.")

    model = _load_model(params["modeling"]["export_path"])
    
    bt, metrics = _backtest_expanding(model, X, y, dates, start=start)

    # If daily, aggregate to monthly for reporting
    if "daily" in params["data"]["master_table_path"]:
        bt["month"] = pd.to_datetime(bt["date"]).dt.to_period("M").dt.to_timestamp()
        bt_monthly = bt.groupby("month", as_index=False).agg({
            "y_true": "sum",
            "y_pred": "sum"
        })
        metrics = _compute_metrics(bt_monthly["y_true"], bt_monthly["y_pred"])
        _plot_performance(bt_monthly, params["modeling"]["performance_fig"])
    else:
        _plot_performance(bt, params["modeling"]["performance_fig"])

    # save artifacts
    bt_path = Path("reports/metrics/backtest_predictions.csv")
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(bt_path, index=False)

    metrics_path = Path("reports/metrics/backtest_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_feature_importance(model, feature_names, params["modeling"]["feature_importance_fig"])

    return {
        "backtest_predictions_csv": str(bt_path),
        "backtest_metrics_json": str(metrics_path),
        "performance_fig": params["modeling"]["performance_fig"],
        "feature_importance_fig": params["modeling"]["feature_importance_fig"],
        "metrics": metrics,
    }

def _plot_performance(bt: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    plt.plot(bt["y_true"].values, label="Real")
    plt.plot(bt["y_pred"].values, label="Predicho")
    plt.title("Backtest (expanding window): Real vs Predicho")
    plt.xlabel("Paso de tiempo (meses en el backtest)")
    plt.ylabel("Ingresos mensuales")
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _plot_feature_importance(model, feature_names: List[str], out_path: str) -> None:
    import numpy as np

    importance = None
    name = model.__class__.__name__

    # Pipeline (ridge)
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "coef_"):
            importance = np.abs(inner.coef_)
            name = inner.__class__.__name__

    if importance is None and hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    if importance is None:
        LOGGER.info("No hay feature importance disponible para este modelo.")
        return

    imp = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(20)

    plt.figure()
    plt.barh(imp.index[::-1], imp.values[::-1])
    plt.title(f"Top 20 Feature Importance ({name})")
    plt.xlabel("Importancia (abs coef o impurity)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

