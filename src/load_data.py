import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR

def load_raw_data():
    """
    Loads all Olist datasets from URLs.
    """
    base_url = "https://raw.githubusercontent.com/paukiss/MIADAS13/refs/heads/main/data/dataset/"
    
    datasets = {
        "orders": "olist_orders_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "translation": "product_category_name_translation.csv",
        "geolocation": "olist_geolocation_dataset.csv"
    }
    
    data = {}
    for name, file_name in datasets.items():
        url = f"{base_url}{file_name}"
        print(f"Loading {name} from {url}...")
        data[name] = pd.read_csv(url)
        
    return data
