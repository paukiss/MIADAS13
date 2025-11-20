import pandas as pd
from src.preprocess import build_master_table, aggregate_monthly
from src.features import create_features
from src.models import train_model, predict_model

def run_monthly_pipeline(data, data_until_month, target_month):
    """
    Simulates the pipeline running at a specific point in time.
    
    Args:
        data: Dictionary of all raw dataframes
        data_until_month: The last month of data available (e.g., '2018-07')
        target_month: The month to predict (e.g., '2018-08')
    """
    print(f"--- Running Pipeline Simulation ---")
    print(f"Current Date (Simulated): End of {data_until_month}")
    print(f"Goal: Predict Revenue for {target_month}")
    
    # 1. Preprocess (Build Master Table + Aggregate)
    # In a real scenario, we would filter raw data by date BEFORE building master table
    # For simplicity here, we build it and then filter, assuming no data leakage in build_master_table
    # (build_master_table uses order_purchase_timestamp, so we can filter the result)
    
    # Note: Ideally we should filter orders/payments/etc BEFORE passing to build_master_table
    # to truly simulate "what we knew then".
    # But build_master_table is complex. Let's filter the aggregated result for now, 
    # assuming the "date" of the record is the order date.
    
    master_table = build_master_table(data, save=False)
    monthly_df = aggregate_monthly(master_table)
    
    # Filter known data
    known_data = monthly_df[monthly_df["year_month"] <= data_until_month].copy()
    
    # 2. Feature Engineering
    # We generate features on known data
    features_df = create_features(known_data)
    
    # 3. Train Model
    # We use all available history to train
    # Drop NaNs caused by lags
    train_df = features_df.dropna()
    
    target_col = "monthly_revenue"
    # Exclude non-feature columns
    exclude_cols = ["year_month", target_col, "month_index", "month", "year", "quarter"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    model = train_model(X_train, y_train, model_type="linear")
    print("Model trained on available history.")
    
    # 4. Predict Next Month
    # We need to construct the feature row for the target_month
    # This is tricky because 'create_features' relies on shift() of existing rows.
    # We can append a dummy row for the target month and re-run create_features
    
    # Create a dummy row with 0s for numeric cols
    dummy_row = pd.DataFrame([known_data.iloc[-1].copy()]) # Copy last row structure
    dummy_row["year_month"] = target_month
    for col in dummy_row.columns:
        if col != "year_month":
            dummy_row[col] = 0 # Unknown values
            
    # Combine for feature generation
    combined = pd.concat([known_data, dummy_row], ignore_index=True)
    combined_features = create_features(combined)
    
    # Get the row for target_month
    target_row = combined_features[combined_features["year_month"] == target_month]
    X_target = target_row[feature_cols]
    
    # Handle NaNs in X_target (if any lag is missing, fill with 0 or mean)
    X_target = X_target.fillna(0)
    
    prediction = predict_model(model, X_target)[0]
    
    print(f"Predicted Revenue for {target_month}: R$ {prediction:,.2f}")
    return prediction
