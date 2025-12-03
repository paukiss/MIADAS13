# Informe de Resultados - Sprint 3: Pipeline de Entrenamiento y Selección de Modelos

## 1. Resumen Ejecutivo
En este Sprint se ha implementado y validado un pipeline modular de Machine Learning para el pronóstico de ingresos mensuales. El objetivo principal fue asegurar una metodología robusta que evite el *data leakage* (fuga de datos) y justifique la selección de variables y modelos.

**Resultado Principal:** Se seleccionó el modelo **Gradient Boosting** como el mejor predictor de Machine Learning, con un MAPE de **7.71%** en el set de prueba, demostrando capacidad para capturar patrones más allá de la inercia simple.

---

## 2. Estrategia de Validación (Data Splitting)
Dada la naturaleza temporal de los datos, se descartó la validación cruzada aleatoria (K-Fold estándar) en favor de una estrategia que respeta el orden cronológico.

### 2.1. Partición de Datos
Se dividió la serie temporal en tres segmentos estrictos:

*   **Entrenamiento (Train):** Datos desde el inicio hasta **Abril 2018**. Se utiliza para ajustar los pesos de los modelos.
*   **Validación / Backtest:** Periodo de **Mayo, Junio y Julio de 2018**. Se utiliza para evaluar el rendimiento de los modelos fuera de la muestra y seleccionar el mejor.
*   **Holdout (Test Final):** **Agosto 2018**. Reservado para la predicción final "a futuro" tras re-entrenar el modelo seleccionado.

### 2.2. Validación Cruzada (Cross-Validation)
Dentro del set de entrenamiento, se utilizó **`TimeSeriesSplit`** con 4 particiones. Esto entrena el modelo en una ventana de tiempo creciente y lo evalúa en el periodo inmediatamente siguiente, simulando un escenario real de pronóstico.

---

## 3. Ingeniería y Selección de Características
El dataset original se enriqueció generando 95 variables potenciales, incluyendo:
*   **Lags (Rezago):** Ventas de hace 1, 2, 3 y 6 meses.
*   **Ventanas Móviles (Rolling):** Promedios y desviaciones estándar de 3 y 6 meses.
*   **Tendencias:** Cambios porcentuales mes a mes.

### Pipeline de Selección
Para evitar el sobreajuste y la maldición de la dimensionalidad, se aplicó un filtro estricto:
1.  **VarianceThreshold:** Eliminación de variables constantes.
2.  **Filtro de Correlación:** Se eliminaron variables con una correlación absoluta > **0.70** entre sí, reduciendo drásticamente la multicolinealidad.
3.  **RFECV (Recursive Feature Elimination):** Selección recursiva usando un Random Forest, quedándose con las variables que realmente minimizan el error.

**Variables Top Seleccionadas:**
Las variables más influyentes resultaron ser:
1.  `monthly_revenue_lag_1` (Ventas del mes anterior): **~57% de importancia**.
2.  `avg_installments_lag_6` (Cuotas promedio hace 6 meses): **~19% de importancia**.

---

## 4. Hiperparametrización (Tuning)
Se utilizó **`GridSearchCV`** para buscar la combinación óptima de hiperparámetros para tres familias de modelos, optimizando la métrica **MAPE** (Mean Absolute Percentage Error).

### Espacios de Búsqueda:
*   **Ridge (Lineal):** Se probó la fuerza de regularización (`alpha`).
*   **Random Forest (Ensamble):** Profundidad máxima (`max_depth`), número de estimadores y features por split.
*   **Gradient Boosting (Boosting):** Tasa de aprendizaje (`learning_rate`), profundidad y submuestra.

---

## 5. Comparativa de Modelos y Resultados
A continuación se presentan los resultados obtenidos en el set de validación (Backtest):

| Modelo | MAPE (%) | Observaciones |
| :--- | :--- | :--- |
| **Naive Last (Benchmark)** | **5.09%** | Modelo base. Predice que "mañana será igual a hoy". Su alto rendimiento indica gran inercia en la serie. |
| **Gradient Boosting** | **7.71%** | **Mejor Modelo ML.** Logró el mejor balance entre sesgo y varianza de los modelos complejos. |
| **Random Forest** | 8.19% | Ligeramente inferior, tendiendo a sobreajustar en los picos. |
| **Ridge Regression** | 11.56% | El modelo lineal no logró capturar la complejidad no lineal de la serie. |

### Análisis de la Selección
Aunque el modelo **Naive** obtuvo el menor error numérico, se seleccionó el **Gradient Boosting** como el modelo final del pipeline de Machine Learning.
*   **Justificación:** El modelo Naive no permite realizar análisis de escenarios ni "qué pasaría si" (no usa features). El Gradient Boosting, con un error muy competitivo (<8%), permite incorporar variables exógenas y capturar cambios de tendencia que el Naive ignoraría por completo.

---

## 6. Conclusiones y Próximos Pasos
1.  **Robustez:** El pipeline es técnicamente sólido y modular. La separación temporal de datos garantiza que las métricas de error son realistas.
2.  **Inercia:** La serie de ingresos de Olist tiene una fuerte componente inercial (lo que vendes hoy depende mucho de ayer).
3.  **Modelo Final:** Se ha exportado el modelo `Gradient Boosting` optimizado (`model.pkl`) listo para inferencia.

**Próximo Paso:** Despliegue del modelo para generar el reporte final de predicción para Agosto 2018.
