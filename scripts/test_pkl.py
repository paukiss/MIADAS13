import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from forecast_income.features.feature_engineering import FeatureConfig, create_features
from forecast_income.utils.config import load_config

def test_model_pkl():
    print(f"🔍 Iniciando prueba con DATOS REALES...")
    
    # 1. Cargar Configuración y Datos Reales
    params = load_config("config/params.yaml")
    master_path = params["data"]["master_table_path"]
    
    print(f"📂 Leyendo datos históricos desde: {master_path}")
    master = pd.read_csv(master_path, parse_dates=[params["data"]["index_col"]])
    master = master.sort_values("month").reset_index(drop=True)
    
    print(f"📅 Último mes en los datos: {master['month'].iloc[-1].date()}")

    # 2. Generar Features (Ingeniería de Características)
    print(f"⚙️ Generando variables (Lags, Rolling means, etc)...")
    cfg = FeatureConfig(
        base_cols=params["features"]["base_cols"],
        lags=params["features"]["lags"],
        rolling_windows=params["features"]["rolling_windows"],
        add_seasonality=params["features"]["add_seasonality"],
        add_trend=params["features"]["add_trend"],
        target_col=params["features"]["target_col"],
    )
    
    # Esto calcula las 95 columnas basadas en la historia real
    df_features, _ = create_features(master, cfg, date_col="month")
    
    # Guardar df_features en un CSV para inspección
    output_csv = "data/processed/features_generated.csv"
    df_features.to_csv(output_csv, index=False)
    print(f"💾 Features generadas guardadas en: {output_csv}")
    
    # Tomamos la ÚLTIMA fila, que tiene los datos necesarios para predecir el SIGUIENTE mes
    real_input_row = df_features.iloc[[-1]]

    # 3. Cargar Modelo y Selección de Features
    model_path = params["modeling"]["export_path"]
    features_path = Path(model_path).parent / "selected_features.json"
    
    print(f"📦 Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    
    print(f"📋 Filtrando las columnas seleccionadas...")
    selected_cols = json.load(open(features_path))
    
    # Filtramos el input para que tenga solo lo que el modelo pide
    X_real = real_input_row[selected_cols]

    print("-" * 50)
    print("🔎 DATOS REALES QUE ENTRAN AL MODELO (Top 5):")
    print(X_real.T) # Mostramos las primeras 10 para que veas valores reales
    print("-" * 50)

    # 4. Predecir
    print(f"🔮 Ejecutando predicción...")
    prediction = model.predict(X_real)[0]
    
    print(f"✅ ¡ÉXITO! Predicción basada en historia real.")
    print(f"💰 Ingreso Predicho para el mes siguiente: R$ {prediction:,.2f}")

if __name__ == "__main__":
    test_model_pkl()
