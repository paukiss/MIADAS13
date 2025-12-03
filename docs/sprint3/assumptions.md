# Supuestos operativos

1) El ingreso mensual se aproxima por `payment_value` (pagos registrados) en el mes del `order_purchase_timestamp`.

2) Se predice 1 mes adelante (horizonte fijo).

3) Los datos nuevos entran por “mes completo” (batch mensual).

4) Se permiten features basadas en lags (no hay fuga si el corte temporal está bien aplicado).
