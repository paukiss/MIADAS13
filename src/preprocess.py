import pandas as pd
import numpy as np
from pathlib import Path
from src.config import MIN_DATE, MAX_DATE, TRAIN_VAL_END, BACKTEST_START, BACKTEST_END, TEST_FINAL

def mode_or_nan(series: pd.Series):
    mode_vals = series.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else np.nan

def mean_length(series: pd.Series):
    non_null = series.dropna().astype(str)
    lengths = non_null.map(len)
    return lengths.mean() if not lengths.empty else np.nan

def build_master_table(data, save=False, output_dir=Path('data/interim')):
    """
    Generates the master table with 60+ features for modeling and business metrics.
    Args:
        data: Dictionary containing all loaded dataframes.
    """
    orders = data['orders'].copy()
    payments = data['payments'].copy()
    items = data['items'].copy()
    customers = data['customers'].copy()
    products = data['products'].copy()
    sellers = data['sellers'].copy()
    reviews = data['reviews'].copy()
    translation = data['translation'].copy()

    items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'], errors='coerce')
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'], errors='coerce')
    reviews['review_answer_timestamp'] = pd.to_datetime(reviews.get('review_answer_timestamp'), errors='coerce')

    payments_agg = (
        payments.groupby('order_id').agg(
            payment_total=pd.NamedAgg(column='payment_value', aggfunc='sum'),
            payment_mean=pd.NamedAgg(column='payment_value', aggfunc='mean'),
            payment_median=pd.NamedAgg(column='payment_value', aggfunc='median'),
            payment_installments_max=pd.NamedAgg(column='payment_installments', aggfunc='max'),
            payment_installments_mean=pd.NamedAgg(column='payment_installments', aggfunc='mean'),
            payment_types=pd.NamedAgg(column='payment_type', aggfunc=lambda x: ','.join(sorted(set(x)))),
            primary_payment_type=pd.NamedAgg(column='payment_type', aggfunc=mode_or_nan),
            payments_count=pd.NamedAgg(column='payment_sequential', aggfunc='count')
        ).reset_index()
    )

    items_agg = (
        items.groupby('order_id').agg(
            total_items=pd.NamedAgg(column='order_item_id', aggfunc='count'),
            distinct_products=pd.NamedAgg(column='product_id', aggfunc=pd.Series.nunique),
            distinct_sellers=pd.NamedAgg(column='seller_id', aggfunc=pd.Series.nunique),
            total_price=pd.NamedAgg(column='price', aggfunc='sum'),
            avg_price=pd.NamedAgg(column='price', aggfunc='mean'),
            max_price=pd.NamedAgg(column='price', aggfunc='max'),
            min_price=pd.NamedAgg(column='price', aggfunc='min'),
            total_freight=pd.NamedAgg(column='freight_value', aggfunc='sum'),
            avg_freight=pd.NamedAgg(column='freight_value', aggfunc='mean'),
            shipping_limit_max=pd.NamedAgg(column='shipping_limit_date', aggfunc='max'),
            shipping_limit_min=pd.NamedAgg(column='shipping_limit_date', aggfunc='min')
        ).reset_index()
    )

    items_products = items.merge(products, on='product_id', how='left')
    items_products = items_products.merge(translation, on='product_category_name', how='left')
    items_products['product_volume_cm3'] = (
        items_products['product_length_cm'] *
        items_products['product_height_cm'] *
        items_products['product_width_cm']
    )

    product_agg = (
        items_products.groupby('order_id').agg(
            top_category_pt=pd.NamedAgg(column='product_category_name', aggfunc=mode_or_nan),
            top_category_en=pd.NamedAgg(column='product_category_name_english', aggfunc=mode_or_nan),
            product_weight_sum=pd.NamedAgg(column='product_weight_g', aggfunc='sum'),
            product_weight_mean=pd.NamedAgg(column='product_weight_g', aggfunc='mean'),
            product_length_mean=pd.NamedAgg(column='product_length_cm', aggfunc='mean'),
            product_height_mean=pd.NamedAgg(column='product_height_cm', aggfunc='mean'),
            product_width_mean=pd.NamedAgg(column='product_width_cm', aggfunc='mean'),
            product_volume_sum=pd.NamedAgg(column='product_volume_cm3', aggfunc='sum'),
            product_volume_mean=pd.NamedAgg(column='product_volume_cm3', aggfunc='mean')
        ).reset_index()
    )

    items_sellers = items.merge(sellers, on='seller_id', how='left')
    seller_agg = (
        items_sellers.groupby('order_id').agg(
            seller_state_mode=pd.NamedAgg(column='seller_state', aggfunc=mode_or_nan),
            seller_city_mode=pd.NamedAgg(column='seller_city', aggfunc=mode_or_nan),
            seller_count=pd.NamedAgg(column='seller_id', aggfunc=pd.Series.nunique)
        ).reset_index()
    )

    reviews_agg = (
        reviews.groupby('order_id').agg(
            review_score_mean=pd.NamedAgg(column='review_score', aggfunc='mean'),
            review_score_std=pd.NamedAgg(column='review_score', aggfunc='std'),
            review_count=pd.NamedAgg(column='review_id', aggfunc='count'),
            review_comment_count=pd.NamedAgg(column='review_comment_message', aggfunc=lambda x: x.notna().sum()),
            review_comment_avg_len=pd.NamedAgg(column='review_comment_message', aggfunc=mean_length),
            review_latest_score=pd.NamedAgg(column='review_score', aggfunc='last'),
            review_latest_date=pd.NamedAgg(column='review_creation_date', aggfunc='max'),
            review_response_rate=pd.NamedAgg(column='review_answer_timestamp', aggfunc=lambda x: x.notna().mean())
        ).reset_index()
    )

    master = (orders
              .merge(customers, on='customer_id', how='left')
              .merge(payments_agg, on='order_id', how='left')
              .merge(items_agg, on='order_id', how='left')
              .merge(product_agg, on='order_id', how='left')
              .merge(seller_agg, on='order_id', how='left')
              .merge(reviews_agg, on='order_id', how='left'))

    master['order_purchase_timestamp'] = pd.to_datetime(master['order_purchase_timestamp'], errors='coerce')
    master['order_approved_at'] = pd.to_datetime(master['order_approved_at'], errors='coerce')
    master['order_delivered_carrier_date'] = pd.to_datetime(master['order_delivered_carrier_date'], errors='coerce')
    master['order_delivered_customer_date'] = pd.to_datetime(master['order_delivered_customer_date'], errors='coerce')
    master['order_estimated_delivery_date'] = pd.to_datetime(master['order_estimated_delivery_date'], errors='coerce')

    master = master.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    master['customer_order_number'] = master.groupby('customer_unique_id').cumcount() + 1
    master['customer_is_repeat'] = (master['customer_order_number'] > 1).astype(int)
    master['days_since_prev_purchase'] = master.groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    master['days_since_first_purchase'] = (
        master['order_purchase_timestamp'] -
        master.groupby('customer_unique_id')['order_purchase_timestamp'].transform('min')
    ).dt.days

    master['delivery_time_days'] = (
        master['order_delivered_customer_date'] - master['order_purchase_timestamp']
    ).dt.days
    master['approval_lag_hours'] = (
        master['order_approved_at'] - master['order_purchase_timestamp']
    ).dt.total_seconds() / 3600
    master['carrier_lag_hours'] = (
        master['order_delivered_carrier_date'] - master['order_approved_at']
    ).dt.total_seconds() / 3600
    master['delivery_delay_vs_estimate'] = (
        master['order_delivered_customer_date'] - master['order_estimated_delivery_date']
    ).dt.days
    master['delivered_on_time'] = (master['delivery_delay_vs_estimate'] <= 0).astype(int)

    master['order_month'] = master['order_purchase_timestamp'].dt.to_period('M').astype(str)
    master['order_week'] = master['order_purchase_timestamp'].dt.to_period('W').astype(str)
    master['weekday_purchase'] = master['order_purchase_timestamp'].dt.dayofweek
    master['hour_purchase'] = master['order_purchase_timestamp'].dt.hour

    master['avg_item_value'] = master['total_price'] / master['total_items']
    master['freight_ratio'] = master['total_freight'] / master['total_price']
    master['installments_ratio'] = master['payment_installments_mean'] / master['payments_count'].replace({0: np.nan})
    master['contribution_margin_proxy'] = master['total_price'] - master['total_freight']
    master.replace([np.inf, -np.inf], np.nan, inplace=True)

    master['is_delivered'] = (master['order_status'] == 'delivered').astype(int)
    master['is_canceled'] = (master['order_status'] == 'canceled').astype(int)
    
    # Filter by date range
    mask = (master["order_purchase_timestamp"] >= MIN_DATE) & (master["order_purchase_timestamp"] <= MAX_DATE)
    master = master.loc[mask].copy()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"master_table_{master['order_month'].max()}.csv"
        master.to_csv(file_path, index=False)
        print(f"Tabla maestra guardada en: {file_path}")

    return master

def aggregate_monthly(master_table):
    """
    Aggregates the master table by month to create the time series dataset.
    """
    # Define aggregation dictionary based on user's preferred variables
    agg_dict = {
        'payment_total': 'sum', # Target: monthly_revenue
        'order_id': 'count', # total_orders
        'payment_installments_max': 'mean',
        'payments_count': 'mean',
        'total_items': 'sum',
        'distinct_products': 'mean',
        'distinct_sellers': 'mean',
        'total_price': 'sum', # Gross sales (similar to revenue but useful)
        'max_price': 'max',
        'min_price': 'min',
        'total_freight': 'sum',
        'review_score_mean': 'mean',
        'review_count': 'sum',
        'review_latest_score': 'mean',
        'delivery_time_days': 'mean',
        'approval_lag_hours': 'mean',
        'carrier_lag_hours': 'mean',
        'delivery_delay_vs_estimate': 'mean',
        'delivered_on_time': 'mean',
        'avg_item_value': 'mean',
        'freight_ratio': 'mean',
        'contribution_margin_proxy': 'sum',
        'is_delivered': 'mean',
        'is_canceled': 'mean'
    }
    
    # Group by order_month (which is year_month)
    monthly_df = master_table.groupby('order_month').agg(agg_dict).reset_index()
    
    # Rename columns for clarity
    monthly_df.rename(columns={
        'order_month': 'year_month',
        'payment_total': 'monthly_revenue',
        'order_id': 'total_orders',
        'total_price': 'gross_sales_amount'
    }, inplace=True)
    
    return monthly_df

def split_data(df):
    """
    Splits the data into Train+Val, Backtest, and Final Test sets.
    """
    # Ensure year_month is comparable
    # We can use string comparison for "YYYY-MM" format
    
    train_val = df[df["year_month"] <= TRAIN_VAL_END[:7]].copy()
    
    backtest_mask = (df["year_month"] >= BACKTEST_START[:7]) & (df["year_month"] <= BACKTEST_END[:7])
    backtest = df[backtest_mask].copy()
    
    final_test = df[df["year_month"] == TEST_FINAL[:7]].copy()
    
    return train_val, backtest, final_test
