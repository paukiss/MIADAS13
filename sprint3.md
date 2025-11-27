# Sprint 3: Forecast de Ingresos (Mensual y Diario)

Este documento resume la metodología, estrategia de validación y configuración del pipeline implementado para el Sprint 3.

## 1. Modos de Ejecución

El sistema soporta dos modos de operación configurables en `config/params.yaml`:

*   **Modo Mensual (`monthly`)**: Predicción directa del ingreso total del próximo mes.
*   **Modo Diario (`daily`)**: Predicción recursiva de los ingresos diarios para los próximos 30 días.

## 2. Estrategia de División de Datos (Split)

Para garantizar una evaluación robusta y evitar el *data leakage*, se ha aplicado estrictamente la siguiente división temporal en ambos modelos:

| Fase | Periodo | Descripción |
| :--- | :--- | :--- |
| **Entrenamiento + Validación** | Inicio - **Abril 2018** | Datos utilizados para Feature Selection y Hyperparameter Tuning. |
| **Backtest (Expanding Window)** | **Mayo, Junio, Julio 2018** | Periodo de prueba simulando producción. El modelo se re-entrena mes a mes. |
| **Predicción Final** | **Agosto 2018** | Horizonte de predicción real (futuro desconocido). |

> **Nota**: En el modo diario, el backtest cubre los mismos meses pero evaluando día a día (92 días en total).

## 3. Metodología de Modelado

Tanto para el modelo mensual como para el diario, se aplica un pipeline riguroso de optimización:

### A. Ingeniería de Características (Feature Engineering)
Se generan variables exógenas y endógenas, incluyendo:
*   **Lags**: Valores pasados de ingresos, órdenes, reviews, etc.
*   **Ventanas Móviles (Rolling)**: Medias, desviaciones estándar, mínimos y máximos móviles.
*   **Tendencia y Estacionalidad**: Componentes de series temporales.

### B. Selección de Características (Feature Optimization)
Antes de entrenar, se ejecuta un proceso de selección automática para reducir ruido y dimensionalidad:
1.  **VarianceThreshold**: Eliminación de variables constantes.
2.  **Filtro de Correlación**: Eliminación de variables altamente correlacionadas (> 0.90).
3.  **RFECV (Recursive Feature Elimination)**: Selección de las mejores variables utilizando Random Forest y validación cruzada temporal.

### C. Ajuste de Hiperparámetros (Hyperparameter Tuning)
Se optimizan los modelos candidatos (`Ridge`, `RandomForest`, `GradientBoosting`) utilizando:
*   **GridSearchCV**: Búsqueda exhaustiva en una grilla de parámetros definidos.
*   **TimeSeriesSplit**: Validación cruzada respetando el orden temporal de los datos.

## 5. Reproducibilidad

El modelo final es **completamente reproducible**. Se ha configurado una semilla aleatoria global (`random_seed: 42`) en `config/params.yaml` que controla:

*   **Inicialización de Modelos**: `Ridge`, `RandomForest` y `GradientBoosting` se instancian con `random_state=42`.
*   **Selección de Features**: El `RandomForestRegressor` utilizado dentro de `RFECV` también fija su semilla.
*   **Entorno**: Se recomienda usar el entorno virtual definido en `requirements.txt` para asegurar versiones consistentes de librerías (`scikit-learn`, `pandas`, `numpy`).

Al ejecutar el pipeline completo (`scripts/run_all.sh`), los resultados (métricas, features seleccionadas y predicciones) serán idénticos en cada ejecución.
