from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from forecast_income.utils.logger import get_logger

LOGGER = get_logger(__name__)

def _to_datetime(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out

def build_master_table_daily(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye una tabla diaria agregando información de órdenes, pagos, reviews e items.
    
    Métricas calculadas:
    - daily_revenue: Suma de payment_value
    - total_orders: Conteo de órdenes
    - avg_review_score: Promedio de review_score
    - total_items: Suma de items vendidos
    - total_freight: Suma de freight_value
    - unique_customers: Cantidad de clientes únicos
    - avg_installments: Promedio de cuotas
    """
    # 1. Preparar Orders
    orders = _to_datetime(data["orders"], ["order_purchase_timestamp"])
    orders = orders.dropna(subset=["order_purchase_timestamp"]).copy()
    
    # Crear columna de fecha (día) para agrupar
    orders["date"] = orders["order_purchase_timestamp"].dt.floor("D")

    # 2. Agregar Payments a nivel de orden
    payments_agg = data["payments"].groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "mean")
    )
    
    # 3. Agregar Reviews a nivel de orden
    reviews_agg = data["reviews"].groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )

    # 4. Agregar Items a nivel de orden
    items_agg = data["items"].groupby("order_id", as_index=False).agg(
        total_items=("order_item_id", "count"),
        total_freight=("freight_value", "sum")
    )

    # 5. Merge de todo hacia orders
    df_merged = orders.merge(payments_agg, on="order_id", how="left")
    df_merged = df_merged.merge(reviews_agg, on="order_id", how="left")
    df_merged = df_merged.merge(items_agg, on="order_id", how="left")
    
    df_merged = df_merged.merge(
        data["customers"][["customer_id", "customer_unique_id"]], 
        on="customer_id", 
        how="left"
    )

    # 6. Agrupación Diaria
    df_daily = df_merged.groupby("date", as_index=False).agg(
        daily_revenue=("payment_value", "sum"),
        total_orders=("order_id", "count"),
        avg_review_score=("review_score", "mean"),
        total_items=("total_items", "sum"),
        total_freight=("total_freight", "sum"),
        unique_customers=("customer_unique_id", "nunique"),
        avg_installments=("payment_installments", "mean")
    )

    # 7. Limpieza y columnas de calendario
    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    
    # Filtrar datos posteriores a 2018-08-22 debido a incompletitud/caída drástica en el dataset original
    # Esto asegura que el modelo entrene y prediga sobre un estado "saludable" del negocio.
    df_daily = df_daily[df_daily["date"] <= "2018-08-22"].copy()
    
    # Rellenar días faltantes (si no hubo ventas en un día, debería aparecer con 0)
    # Creamos un rango completo de fechas
    full_idx = pd.date_range(start=df_daily["date"].min(), end=df_daily["date"].max(), freq="D")
    df_daily = df_daily.set_index("date").reindex(full_idx).reset_index().rename(columns={"index": "date"})
    
    # Rellenar NaNs con 0 para métricas de volumen
    fill_zeros = ["daily_revenue", "total_orders", "total_items", "total_freight", "unique_customers"]
    df_daily[fill_zeros] = df_daily[fill_zeros].fillna(0)
    
    # Rellenar NaNs con forward fill o media para promedios (o dejarlos NaN si el modelo lo maneja, pero mejor imputar)
    # Para review score e installments, si no hubo ventas, no hay score. 
    # Podemos usar ffill para mantener el "estado" del mercado, o 0.
    # Usaremos ffill y luego bfill para los iniciales.
    df_daily["avg_review_score"] = df_daily["avg_review_score"].ffill().bfill()
    df_daily["avg_installments"] = df_daily["avg_installments"].ffill().bfill()

    df_daily["year"] = df_daily["date"].dt.year
    df_daily["month"] = df_daily["date"].dt.month
    df_daily["day"] = df_daily["date"].dt.day
    df_daily["day_of_week"] = df_daily["date"].dt.dayofweek
    
    return df_daily

def save_daily_table(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
