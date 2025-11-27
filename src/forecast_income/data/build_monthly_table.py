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

def build_master_table_monthly(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye una tabla mensual agregando información de órdenes, pagos, reviews e items.
    
    Métricas calculadas:
    - monthly_revenue: Suma de payment_value
    - total_orders: Conteo de órdenes
    - avg_review_score: Promedio de review_score
    - active_days: Días distintos con ventas en el mes
    - total_items: Suma de items vendidos
    - total_freight: Suma de freight_value
    - unique_customers: Cantidad de clientes únicos
    - avg_installments: Promedio de cuotas
    """
    # 1. Preparar Orders
    orders = _to_datetime(data["orders"], ["order_purchase_timestamp"])
    # Filtrar fechas nulas si las hay
    orders = orders.dropna(subset=["order_purchase_timestamp"]).copy()
    
    # Crear columna de mes (inicio de mes) para agrupar
    orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    # Crear columna de fecha (día) para active_days
    orders["order_date"] = orders["order_purchase_timestamp"].dt.date

    # 2. Agregar Payments a nivel de orden (evita duplicados si mergeamos directo)
    # Un pedido puede tener múltiples pagos, los sumamos.
    payments_agg = data["payments"].groupby("order_id", as_index=False).agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "mean")
    )
    
    # 3. Agregar Reviews a nivel de orden
    # Un pedido puede tener múltiples reviews (raro, pero posible), promediamos.
    reviews_agg = data["reviews"].groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean")
    )

    # 4. Agregar Items a nivel de orden
    items_agg = data["items"].groupby("order_id", as_index=False).agg(
        total_items=("order_item_id", "count"),
        total_freight=("freight_value", "sum")
    )

    # 5. Merge de todo hacia orders
    # Usamos left join para mantener todas las órdenes, aunque no tengan pago/review/items (casos raros)
    df_merged = orders.merge(payments_agg, on="order_id", how="left")
    df_merged = df_merged.merge(reviews_agg, on="order_id", how="left")
    df_merged = df_merged.merge(items_agg, on="order_id", how="left")
    
    # Merge con customers para obtener customer_unique_id
    df_merged = df_merged.merge(
        data["customers"][["customer_id", "customer_unique_id"]], 
        on="customer_id", 
        how="left"
    )

    # --- Logic for Top Category ---
    # Merge items with products to get category
    items_with_cat = data["items"].merge(data["products"][["product_id", "product_category_name"]], on="product_id", how="left")
    # Merge with orders to get date
    items_with_cat = items_with_cat.merge(orders[["order_id", "month"]], on="order_id", how="inner")
    # Group by month and category, count items
    cat_counts = items_with_cat.groupby(["month", "product_category_name"]).size().reset_index(name="count")
    # Sort and take top 1
    top_cats = cat_counts.sort_values(["month", "count"], ascending=[True, False]).drop_duplicates("month")
    top_cats = top_cats[["month", "product_category_name"]].rename(columns={"product_category_name": "top_category"})

    # --- Logic for New Sellers ---
    # Merge items with orders to get date
    items_with_date = data["items"].merge(orders[["order_id", "order_purchase_timestamp"]], on="order_id", how="inner")
    # Find first appearance of each seller
    seller_first_seen = items_with_date.groupby("seller_id")["order_purchase_timestamp"].min().dt.to_period("M").dt.to_timestamp().reset_index(name="month")
    # Count new sellers per month
    new_sellers = seller_first_seen.groupby("month").size().reset_index(name="new_sellers")

    # 6. Agrupación Mensual
    df_monthly = df_merged.groupby("month", as_index=False).agg(
        monthly_revenue=("payment_value", "sum"),
        total_orders=("order_id", "count"),
        avg_review_score=("review_score", "mean"),
        active_days=("order_date", "nunique"),
        total_items=("total_items", "sum"),
        total_freight=("total_freight", "sum"),
        unique_customers=("customer_unique_id", "nunique"),
        avg_installments=("payment_installments", "mean")
    )

    # Merge new metrics
    df_monthly = df_monthly.merge(top_cats, on="month", how="left")
    df_monthly = df_monthly.merge(new_sellers, on="month", how="left")
    df_monthly["new_sellers"] = df_monthly["new_sellers"].fillna(0).astype(int)

    # 7. Limpieza y columnas de calendario
    df_monthly = df_monthly.sort_values("month").reset_index(drop=True)
    
    # Filtrar meses incompletos/vacíos al final
    # El dataset tiene datos hasta agosto 2018, pero agosto está incompleto (hasta el 22).
    # Para un modelo mensual robusto, mejor cortar en Julio 2018.
    df_monthly = df_monthly[df_monthly["month"] < "2018-08-01"].copy()
    
    df_monthly["year"] = df_monthly["month"].dt.year
    df_monthly["month_num"] = df_monthly["month"].dt.month

    # Rellenar posibles NaNs
    for c in ["avg_review_score", "avg_installments"]:
        if c in df_monthly.columns:
            df_monthly[c] = df_monthly[c].fillna(df_monthly[c].median())
            
    for c in ["total_items", "total_freight", "unique_customers", "total_orders"]:
        if c in df_monthly.columns:
            df_monthly[c] = df_monthly[c].fillna(0)

    return df_monthly

def save_master_table(master: pd.DataFrame, out_path: str) -> None:
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(dest, index=False)
    LOGGER.info("Master table guardada en %s", dest)
