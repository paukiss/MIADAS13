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
    
    # 2. Correlación > 0.90
    # Calculamos matriz de correlación absoluta
    corr_matrix = X_train.corr().abs()
    # Seleccionamos triángulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Buscamos columnas con correlación > 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    X_train = X_train.drop(columns=to_drop)
    LOGGER.info(f"Features tras filtro correlación (>0.90): {X_train.shape[1]}")
    LOGGER.info(f"Columnas eliminadas por correlación: {len(to_drop)}")
    
    # 3. RFECV
    # Usamos RandomForestRegressor como estimador base
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    # TimeSeriesSplit para validación cruzada (respetando orden temporal)
    # n_splits=3 como solicitado. Si hay pocos datos, esto puede fallar o ser muy pequeño.
    # Ajustamos min_splits si es necesario, pero intentamos 3.
    n_splits = 3
    if len(X_train) < 5: # Muy pocos datos para 3 splits
        n_splits = 2
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        min_features_to_select=5, # Al menos quedarnos con algo
        n_jobs=-1
    )
    
    rfecv.fit(X_train, y_train)
    
    selected_cols = X_train.columns[rfecv.support_].tolist()
    X_train_selected = X_train[selected_cols]
    
    LOGGER.info(f"Features tras RFECV: {len(selected_cols)}")
    
    # Ranking de importancia (solo de las seleccionadas)
    # RFECV no tiene feature_importances_ directo del mejor modelo re-entrenado en todo X_train,
    # pero podemos ajustar un RF final en las features seleccionadas para ver importancia.
    rf_final = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_final.fit(X_train_selected, y_train)
    
    importances = pd.Series(rf_final.feature_importances_, index=selected_cols).sort_values(ascending=False)
    
    LOGGER.info("Top 20 Features más importantes:")
    LOGGER.info("\n" + str(importances.head(20)))
    
    return X_train_selected, selected_cols
