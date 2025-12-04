import pandas as pd

orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
merged = orders.merge(payments, on="order_id")

august_2018 = merged[
    (merged["order_purchase_timestamp"].dt.year == 2018) & 
    (merged["order_purchase_timestamp"].dt.month == 8)
]

print(f"Total Revenue August 2018: {august_2018['payment_value'].sum()}")
print(f"Max Date in August: {august_2018['order_purchase_timestamp'].max()}")
