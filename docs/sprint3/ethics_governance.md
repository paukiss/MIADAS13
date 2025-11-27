# Ética & Gobernanza — Sprint 3

## Privacidad
Se evita exponer datos personales: el modelo opera sobre **agregados mensuales**.

## Sesgos
Sesgos típicos posibles:
- cambios en mix de productos/categorías,
- estacionalidad no estable,
- ventas anómalas por promociones.

## Trazabilidad
- Versionar: `config/params.yaml`, `models/final/metadata.json`, métricas en `reports/metrics/`.
- Reproducibilidad: caché de datasets en `data/raw/`.

## Monitoreo recomendado (Sprint 4)
- Deriva de datos: cambios en distribución de `total_orders`, `unique_customers`.
- Drift de performance: MAPE mensual.
