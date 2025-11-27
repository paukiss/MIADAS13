# Model Card — Forecast de Ingresos Mensuales (Grupo 10)

## Modelo
Regressor seleccionado por menor MAPE en holdout y backtest.

## Datos
Olist Brazilian E-Commerce Dataset (2016–2018). Se agregan transacciones a frecuencia mensual.

## Target
`monthly_revenue_next`: ingresos del próximo mes.

## Uso previsto
Forecast de planificación operativa (cashflow e inventario) a nivel de negocio (no individual).

## Limitaciones
- Dataset histórico (hasta 2018).
- No incluye variables externas (economía, campañas, shocks).
- Serie relativamente corta (pocos puntos mensuales), lo que limita modelos complejos.

## Riesgos / ética
Riesgo bajo en términos de privacidad porque se trabaja con agregados mensuales.
