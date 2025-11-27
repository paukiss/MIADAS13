from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from forecast_income.utils.logger import get_logger

LOGGER = get_logger(__name__)

DATASETS = {
    "orders": "olist_orders_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

def download_raw_datasets(base_url: str, cache_dir: str = "data/raw") -> None:
    """Descarga datasets desde el repo (raw) y los guarda en cache_dir."""
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, file_name in DATASETS.items():
        url = f"{base_url}{file_name}"
        LOGGER.info("Descargando %s desde %s", name, url)
        df = pd.read_csv(url)
        df.to_csv(out_dir / file_name, index=False)

def load_raw_data(base_url: str, cache_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """Carga datasets, priorizando caché local. Si no existe, lee desde URL."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    data: Dict[str, pd.DataFrame] = {}
    for name, file_name in DATASETS.items():
        local_path = cache / file_name
        if local_path.exists():
            LOGGER.info("Leyendo %s desde caché: %s", name, local_path)
            data[name] = pd.read_csv(local_path)
        else:
            url = f"{base_url}{file_name}"
            LOGGER.info("Leyendo %s desde URL: %s", name, url)
            df = pd.read_csv(url)
            data[name] = df
            # cachea para reproducibilidad
            df.to_csv(local_path, index=False)

    return data
