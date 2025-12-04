from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit

from forecast_income.utils.logger import get_logger

LOGGER = get_logger(__name__)

def select_features(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pipeline estricto de selección de características:
    1. VarianceThreshold(0): Elimina constantes.
    2. Correlación > 0.90: Elimina una de las dos variables colineales.
    3. RFECV: Selección recursiva con RandomForest y TimeSeriesSplit.
    
    Retorna:
        X_train_selected: DataFrame con las columnas seleccionadas.
        selected_features: Lista de nombres de las features seleccionadas.
    """
    LOGGER.info(f"Inicio selección de features. Shape original: {X_train.shape}")
    
    # 1. VarianceThreshold
    selector = VarianceThreshold(threshold=0)
    selector.fit(X_train)
    # Get columns to keep
    cols_variance = X_train.columns[selector.get_support()].tolist()
    X_train = X_train[cols_variance]
    LOGGER.info(f"Features tras VarianceThreshold: {X_train.shape[1]}")
    
    # 2. Correlación > 0.70 (Muy estricto para garantizar independencia)
    # Calculamos matriz de correlación absoluta
    corr_matrix = X_train.corr().abs()
    # Seleccionamos triángulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Buscamos columnas con correlación > 0.70
    to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
    
    X_train = X_train.drop(columns=to_drop)
    LOGGER.info(f"Features tras filtro correlación (>0.70): {X_train.shape[1]}")
    LOGGER.info(f"Columnas eliminadas por correlación: {len(to_drop)}")
    
    # 3. RFECV
    # Usamos RandomForestRegressor como estimador base
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    
    LOGGER.info("Ejecutando RFECV (Recursive Feature Elimination)... esto puede tardar.")
    tscv = TimeSeriesSplit(n_splits=2)
    # min_features_to_select=10 para asegurar que no nos quedamos sin nada
    # step=5 para acelerar el proceso
    selector = RFECV(estimator=rf, step=5, cv=tscv, scoring="neg_mean_absolute_percentage_error", n_jobs=-1, min_features_to_select=10)
    selector.fit(X_train, y_train)
    
    selected_cols = X_train.columns[selector.support_].tolist()
    
    # Force max 20 features if RFECV selected more
    if len(selected_cols) > 20:
        LOGGER.info(f"RFECV seleccionó {len(selected_cols)} features. Limitando a las Top 20.")
        # Re-fit to get feature importances on the selected subset
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        rf_temp.fit(X_train[selected_cols], y_train)
        
        importances_temp = pd.Series(rf_temp.feature_importances_, index=selected_cols).sort_values(ascending=False)
        selected_cols = importances_temp.head(20).index.tolist()
        
    X_train_selected = X_train[selected_cols]
    
    LOGGER.info(f"Features seleccionadas finales: {len(selected_cols)}")
    
    # Ranking de importancia (solo de las seleccionadas)
    rf_final = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_final.fit(X_train_selected, y_train)
    
    importances = pd.Series(rf_final.feature_importances_, index=selected_cols).sort_values(ascending=False)
    
    LOGGER.info("Top 20 Features más importantes:")
    LOGGER.info("\n" + str(importances.head(20)))
    
    return X_train_selected, selected_cols
