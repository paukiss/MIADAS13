from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
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

def _split_train_test(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    if test_size <= 0:
        raise ValueError("test_size debe ser > 0")
    split = len(X) - test_size
    if split <= 0:
        raise ValueError("No hay suficientes datos para holdout")

    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:], dates.iloc[:split], dates.iloc[split:]

def _score(y_true, y_pred, dates=None) -> Dict[str, float]:
    # Filter out Trucker's Strike (May 21 - May 31, 2018) if dates provided
    if dates is not None:
        mask = ~((dates >= "2018-05-21") & (dates <= "2018-05-31"))
        if (~mask).sum() > 0:
            # LOGGER.info(f"Excluding {(~mask).sum()} days due to Trucker's Strike")
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }

import matplotlib.pyplot as plt
import seaborn as sns

def _save_plots(X: pd.DataFrame, y: pd.Series, output_dir: str):
    """Genera y guarda gráficos de importancia y correlación de las features seleccionadas."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance (usando RF genérico para comparabilidad)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importances.head(20), x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance (Variables Seleccionadas)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matriz de Correlación (Variables Seleccionadas)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

def perform_tuning(X_train, y_train, X_test, y_test, dates_test, params):
    """
    Ejecuta el GridSearch para los candidatos definidos y evalúa en test.
    Retorna (results_list, best_model_info)
    """
    results: List[Dict[str, Any]] = []

    # Baselines (sin features, solo y)
    y_train_full = y_train.copy().reset_index(drop=True)
    y_test_full = y_test.copy().reset_index(drop=True)
    dates_test_full = dates_test.copy().reset_index(drop=True)

    naive = NaiveLastValue().fit(y_train_full)
    preds_naive = naive.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
    results.append({"model": "naive_last", **_score(y_test_full, preds_naive, dates_test_full)})

    if "seasonal_naive_12" in params["modeling"]["candidates"]:
        seasonal = SeasonalNaive(season_lag=12).fit(y_train_full)
        preds_seasonal = seasonal.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
        results.append({"model": "seasonal_naive_12", **_score(y_test_full, preds_seasonal, dates_test_full)})
        
    if "seasonal_naive_7" in params["modeling"]["candidates"]:
        seasonal = SeasonalNaive(season_lag=7).fit(y_train_full)
        preds_seasonal = seasonal.predict(pd.concat([y_train_full, y_test_full[:-1]]), steps=len(y_test_full))
        results.append({"model": "seasonal_naive_7", **_score(y_test_full, preds_seasonal, dates_test_full)})

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
        sc = _score(y_test, preds, dates_test)
        
        LOGGER.info(f"   ✅ Resultado {name}: MAPE Test={sc['MAPE']:.2f}% | CV Score (Mean MAPE)={-search.best_score_:.2f}%")
        LOGGER.info(f"   ℹ️ Mejores Parámetros: {search.best_params_}")

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
            
    return results, {
        "best_model_name": best_model_name,
        "best_estimator": best_estimator,
        "best_mape": best_mape,
        "best_meta": best_meta
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

    # --- Guardar Features Generadas para Inspección ---
    features_path = Path("data/processed/features_generated.csv")
    # Unimos X con y y dates para tener el dataset completo
    full_dataset = X.copy()
    full_dataset["target"] = y
    full_dataset["date_index"] = dates
    full_dataset.to_csv(features_path, index=False)
    LOGGER.info(f"Features generadas ({X.shape[1]} columnas) guardadas en {features_path}")
    # ------------------------------------------------

    # Timeline Configuration
    train_end = params["modeling"].get("train_end_date")
    backtest_end = params["modeling"].get("backtest_end_date")
    
    if train_end and backtest_end:
        LOGGER.info(f"Configuración de fases detectada: Train hasta {train_end}, Backtest hasta {backtest_end}")
        
        # Define cutoff for later use (logging)
        cutoff = backtest_end
        
        # Ensure dates are datetime
        dates = pd.to_datetime(dates)
        
        # 1. Filter total scope (up to backtest_end)
        mask_scope = dates <= backtest_end
        X = X[mask_scope]
        y = y[mask_scope]
        dates = dates[mask_scope]
        
        # 2. Split Train / Test (Backtest)
        mask_train = dates <= train_end
        X_train = X[mask_train]
        y_train = y[mask_train]
        dates_train = dates[mask_train]
        
        X_test = X[~mask_train]
        y_test = y[~mask_train]
        dates_test = dates[~mask_train]
        
    else:
        # Legacy / Default behavior
        cutoff = params["modeling"].get("train_cutoff_date")
        if cutoff:
            LOGGER.info(f"Filtrando datos de entrenamiento hasta {cutoff}")
            mask = dates <= cutoff
            X = X[mask]
            y = y[mask]
            dates = dates[mask]

        X_train, X_test, y_train, y_test, dates_train, dates_test = _split_train_test(X, y, dates, test_size)

    # --- Feature Selection ---
    LOGGER.info("Ejecutando pipeline de selección de características...")
    X_train, selected_cols = select_features(X_train, y_train, random_state=params["modeling"]["random_seed"])
    X_test = X_test[selected_cols]
    LOGGER.info(f"Features finales para entrenamiento: {len(selected_cols)}")
    
    # Guardar gráficos de justificación
    _save_plots(X_train, y_train, "reports/figures")
    # -------------------------

    results, best_info = perform_tuning(X_train, y_train, X_test, y_test, dates_test, params)
    
    best_estimator = best_info["best_estimator"]
    best_model_name = best_info["best_model_name"]
    best_mape = best_info["best_mape"]
    best_meta = best_info["best_meta"]

    # Export best estimator
    if best_estimator is None:
        raise RuntimeError("No se entrenó ningún modelo candidato (revisa config).")

    # --- RETRAIN ON FULL DATA (Train + Backtest) for Final Prediction ---
    # The user wants the model to be ready for August prediction.
    # So we retrain the best configuration on X (which includes May, Jun, Jul).
    LOGGER.info(f"🔄 Re-entrenando el mejor modelo ({best_model_name}) con TODOS los datos (hasta {cutoff}) para predicción final...")
    
    # Prepare full dataset with selected features
    X_full = X[selected_cols]
    y_full = y
    
    # Apply Log Transform if used in tuning
    if best_meta.get("target_transform") == "log1p":
        y_full = np.log1p(y_full)
        
    # Clone and fit
    from sklearn.base import clone
    final_model = clone(best_estimator)
    final_model.fit(X_full, y_full)
    
    export_path = Path(params["modeling"]["export_path"])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, export_path)
    LOGGER.info(f"✅ Modelo final guardado en {export_path}")

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
