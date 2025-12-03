# Plan de Reentrenamiento Mensual (Retraining Plan)

## 1. Objetivo
Garantizar que el modelo de predicción de ingresos se mantenga actualizado frente a la evolución del negocio, incorporando nuevos datos transaccionales y adaptándose a cambios de tendencia o estacionalidad.

## 2. Frecuencia de Ejecución
El pipeline de reentrenamiento debe ejecutarse **mensualmente**, idealmente el día 1 o 2 de cada mes, una vez que los datos del mes anterior estén consolidados.

## 3. Flujo de Trabajo (Pipeline)

### Paso 1: Ingesta de Datos (Data Ingestion)
*   **Acción**: Descargar los últimos dumps de la base de datos transaccional (Orders, Payments, Items, etc.).
*   **Validación**: Verificar que no existan días faltantes en el mes recién cerrado.
*   **Script**: `python -m src.cli build-master`

### Paso 2: Actualización de Features (Feature Engineering)
*   **Acción**: Recalcular la tabla maestra mensual.
*   **Lógica**:
    *   Generar nuevos lags (t-1, t-2...) con el dato real del mes cerrado.
    *   Actualizar ventanas móviles (rolling mean/std).
    *   Recalcular componentes de tendencia y estacionalidad.

### Paso 3: Reentrenamiento del Modelo (Model Retraining)
*   **Estrategia**:
    *   **Mensual**: Reentrenar el modelo ganador (ej. Ridge) con todo el histórico disponible (Expanding Window).
    *   **Trimestral**: Ejecutar nuevamente la búsqueda de hiperparámetros (`GridSearchCV`) para ajustar la regularización o profundidad de los árboles.
*   **Script**: `python -m src.cli tune`

### Paso 4: Evaluación y Monitoreo (Evaluation)
*   **Acción**: Ejecutar backtest sobre los últimos 3 meses conocidos.
*   **KPIs de Control**:
    *   **MAPE**: Si supera el 15%, activar alerta de revisión manual.
    *   **Bias**: Si el sesgo es consistentemente positivo/negativo por >2 meses, revisar cambios estructurales en el negocio.
    *   **Forecast Realization Rate**: Debe mantenerse entre 90% y 110%.

### Paso 5: Despliegue (Deployment)
*   **Artefacto**: Exportar `models/final/model.pkl` y `models/final/metadata.json`.
*   **Versionado**: Etiquetar el modelo con la fecha de corte (ej. `model_2018_08.pkl`) y subir a almacenamiento de artefactos (MLflow / S3).

## 4. Criterios de Drift (Re-Tuning Trigger)
Se debe ejecutar un **Full Tuning** (búsqueda de nuevo algoritmo o hiperparámetros) si:
1.  **Performance Drift**: El MAPE promedio móvil (3 meses) aumenta un 30% respecto a la línea base.
2.  **Data Drift**: Cambios significativos en la distribución de variables clave (ej. Ticket Promedio cambia drásticamente por una nueva política de precios).
3.  **Business Change**: Entrada a nuevos mercados o categorías que alteren la estructura de ingresos.
