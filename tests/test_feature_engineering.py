import pandas as pd
from forecast_income.features.feature_engineering import FeatureConfig, make_supervised_monthly

def test_make_supervised_shapes():
    master = pd.DataFrame({
        "month": pd.date_range("2017-01-01", periods=24, freq="MS"),
        "monthly_revenue": range(24),
        "total_orders": [10]*24,
        "total_items": [20]*24,
        "total_freight": [5]*24,
        "unique_customers": [7]*24,
        "avg_installments": [2]*24,
        "avg_review_score": [4]*24,
        "year": [2017]*12 + [2018]*12,
        "month_num": list(range(1,13))*2,
    })
    cfg = FeatureConfig(
        base_cols=["monthly_revenue","total_orders","total_items","total_freight","unique_customers","avg_installments","avg_review_score"],
        lags=[1,2,3,6,12],
        rolling_windows=[3,6,12],
        add_seasonality=True,
        add_trend=True,
        target_col="monthly_revenue_next",
    )
    X, y, feats = make_supervised_monthly(master, cfg)
    assert len(X) == len(y)
    assert len(feats) == X.shape[1]
    assert X.shape[0] > 0
