import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class NaiveModel:
    def __init__(self):
        self.last_value = None

    def fit(self, X, y):
        # In a time series context, "fit" for naive might just mean storing the last seen value
        # But typically Naive Forecast for time t is y_{t-1}.
        # If we are predicting a sequence, we need the previous values.
        # Here we assume X contains 'revenue_lag_1' which IS the naive prediction.
        pass

    def predict(self, X):
        # If X has 'monthly_revenue_lag_1', return it.
        if "monthly_revenue_lag_1" in X.columns:
            return X["monthly_revenue_lag_1"].values
        else:
            raise ValueError("Naive model requires 'monthly_revenue_lag_1' feature.")

def train_model(X_train, y_train, model_type="linear"):
    """
    Trains a regression model.
    """
    # Drop NaNs from X_train/y_train if any (due to lags)
    # Align indices
    combined = pd.concat([X_train, y_train], axis=1).dropna()
    X_clean = combined[X_train.columns]
    y_clean = combined[y_train.name]
    
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.fit(X_clean, y_clean)
    return model

def predict_model(model, X):
    """
    Predicts using the trained model.
    """
    # Handle NaNs in X if necessary, but usually we want to predict for all rows
    # If X has NaNs (e.g. first few rows), prediction might fail or return NaN depending on model
    # For sklearn, we usually need to fill or drop. 
    # Here we assume X is prepared.
    
    # Simple fill for now if needed, or let it crash to reveal issues
    X_clean = X.fillna(0) # DANGEROUS for time series, but okay for MVP if lags are missing at start
    
    return model.predict(X_clean)
