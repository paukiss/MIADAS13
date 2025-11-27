from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from forecast_income.features.feature_engineering import FeatureConfig, make_supervised_monthly, make_supervised_daily
from forecast_income.features.selection import select_features
from forecast_income.models.baselines import NaiveLastValue, SeasonalNaive
from forecast_income.models.metrics import mae, rmse, mape, bias
from forecast_income.models.modeling import build_candidates
from forecast_income.utils.logger import get_logger
from forecast_income.utils.config import load_config

LOGGER = get_logger(__name__)

def _load_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    elif "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
        return df.sort_values("month").reset_index(drop=True)
    return df

def _split_train_test(X: pd.DataFrame, y: pd.Series, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_size <= 0:
        raise ValueError("test_size debe ser > 0")
    split = len(X) - test_size
    if split <= 0:
        raise ValueError("No hay suficientes datos para holdout")

    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def _score(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }

def tune_and_select(params_path: str = "config/params.yaml") -> Dict[str, Any]:
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

    if "daily" in params["data"]["master_table_path"]:
        X, y, feature_names, dates = make_supervised_daily(master, cfg)
        test_size = params["modeling"].get("test_size", 180) # Used for internal validation split
    else:
        X, y, feature_names, dates = make_supervised_monthly(master, cfg)
        test_size = params["modeling"].get("test_size", 6) # Used for internal validation split

    # Filter Train + Validation period (cutoff)
    cutoff = params["modeling"].get("train_cutoff_date")
    if cutoff:
        LOGGER.info(f"Filtrando datos de entrenamiento hasta {cutoff}")
        mask = dates <= cutoff
        X = X[mask]
        y = y[mask]
        dates = dates[mask]

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size)

    # --- Feature Selection ---
    LOGGER.info("Ejecutando pipeline de selección de características...")
    X_train, selected_cols = select_features(X_train, y_train, random_state=params["modeling"]["random_seed"])
    X_test = X_test[selected_cols]
    LOGGER.info(f"Features finales para entrenamiento: {len(selected_cols)}")
    # -------------------------

    results: List[Dict[str, Any]] = []

    # Baselines (sin features, solo y)
    y_train_full = y_train.copy().reset_index(drop=True)
    y_test_full = y_test.copy().reset_index(drop=True)

    naive = NaiveLastValue().fit(y_train_full)
    preds_naive = naive.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
    results.append({"model": "naive_last", **_score(y_test_full, preds_naive)})

    if "seasonal_naive_12" in params["modeling"]["candidates"]:
        seasonal = SeasonalNaive(season_lag=12).fit(y_train_full)
        preds_seasonal = seasonal.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
        results.append({"model": "seasonal_naive_12", **_score(y_test_full, preds_seasonal)})
        
    if "seasonal_naive_7" in params["modeling"]["candidates"]:
        seasonal = SeasonalNaive(season_lag=7).fit(y_train_full)
        preds_seasonal = seasonal.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
        results.append({"model": "seasonal_naive_7", **_score(y_test_full, preds_seasonal)})

    # ML candidates with TimeSeries CV
    candidates = build_candidates(random_seed=params["modeling"]["random_seed"])
    tscv = TimeSeriesSplit(n_splits=params["modeling"]["cv_splits"])

    best_model_name = None
    best_estimator = None
    best_mape = float("inf")
    best_meta = {}

    for name in params["modeling"]["candidates"]:
        if name in ["naive_last", "seasonal_naive_12", "seasonal_naive_7"]:
            continue
        cand = candidates.get(name)
        if not cand:
            LOGGER.info("🔎 Candidato no implementado: %s (se omite)", name)
            continue

        LOGGER.info("⚙️ Tuning: %s", name)
        search = GridSearchCV(
            estimator=cand.estimator,
            param_grid=cand.search_space or {},
            cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        preds = best.predict(X_test)
        sc = _score(y_test, preds)
        sc_row = {"model": name, "best_params": search.best_params_, **sc}
        results.append(sc_row)

        if sc["MAPE"] < best_mape:
            best_mape = sc["MAPE"]
            best_model_name = name
            best_estimator = best
            best_meta = {
                "best_params": search.best_params_,
                "cv_best_score_neg_mape": float(search.best_score_),
            }

    # Export best estimator
    if best_estimator is None:
        raise RuntimeError("No se entrenó ningún modelo candidato (revisa config).")

    export_path = Path(params["modeling"]["export_path"])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, export_path)

    # Save selected features
    selected_features_path = export_path.parent / "selected_features.json"
    selected_features_path.write_text(json.dumps(selected_cols, indent=2), encoding="utf-8")

    metadata = {
        "project": params["project"],
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "best_model": best_model_name,
        "best_mape": best_mape,
        "feature_names": selected_cols,
        **best_meta,
    }
    Path(params["modeling"]["export_metadata_path"]).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save metrics summary
    metrics_df = pd.DataFrame(results).sort_values("MAPE")
    Path(params["modeling"]["metrics_csv"]).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(params["modeling"]["metrics_csv"], index=False)

    return {
        "best_model": best_model_name,
        "best_mape": best_mape,
        "export_path": str(export_path),
        "metrics_csv": params["modeling"]["metrics_csv"],
    }
