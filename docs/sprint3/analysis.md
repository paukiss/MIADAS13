# Análisis — Sprint 3 (Producto 3): Forecast de Ingresos Mensuales

## 1) Problema de negocio
Predecir los **ingresos mensuales** de Olist para mejorar:
- planificación financiera (cashflow),
- presupuesto de marketing,
- planificación logística e inventario.

**Horizonte:** 1 mes adelante (t → t+1).  
**Granularidad:** mensual.

## 2) Definición de Target
- `monthly_revenue` = suma de `payment_value` por mes, usando `order_purchase_timestamp`.
- `monthly_revenue_next` = `monthly_revenue` desplazado 1 mes hacia adelante (target supervisado).

## 3) Variables/Features (alto nivel)
Se construye una **master table mensual** con:
- `monthly_revenue` (base)
- `total_orders` (nunique order_id)
- `total_items` (conteo de items)
- `total_freight` (suma freight_value)
- `unique_customers` (nunique customer_unique_id)
- `avg_installments` (promedio payment_installments)
- `avg_review_score` (promedio review_score)

De esas variables agregadas se generan features de series temporales:
- **Lags**: 1,2,3,6,12 meses
- **Medias móviles**: 3,6,12
- **Seasonality**: month_sin/month_cos
- **Trend**: month_index

## 4) Estrategia de evaluación
- Holdout final: últimos **N meses** (`test_months`) del dataset supervisado.
- CV: `TimeSeriesSplit` (sin shuffling).
- Backtest extra: **expanding window** para simular reentrenamiento mensual.

## 5) Modelos comparados
Baselines:
- Naive: y(t+1) = y(t)
- Seasonal Naive (12): y(t+1) = y(t-11) aproximado (misma época año anterior)

Modelos ML:
- Ridge (con escalado)
- Random Forest
- Gradient Boosting

## 6) Métricas (técnicas + lectura de negocio)
- **MAPE**: métrica principal para comunicar precisión (error porcentual).
- **MAE / RMSE**: error absoluto y penalización cuadrática.
- **Bias**: sesgo promedio (positivo = sobrestima; negativo = subestima).

## 7) Entregables Sprint 3 (checklist)
- `models/final/model.pkl` (modelo elegido)
- `reports/metrics/metrics_summary.csv` (comparación)
- `reports/metrics/backtest_*` (predicciones y métricas)
- `reports/figures/feature_importance.png`
- `reports/figures/backtest_performance.png`
