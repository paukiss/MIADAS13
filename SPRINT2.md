# Sprint 2 – Forecast de Ingresos Mensuales (Tema 10)

## 1. Contexto

Trabajamos con los datos de un e-commerce (Olist) para **predecir los ingresos mensuales**.  
El profesor definió que debemos respetar la **temporalidad** y realizar un **backtesting**:

- Período aproximado: de **2016-10** a **2018-08**.
- Usar ~**19–20 meses** para entrenamiento/validación.
- Reservar **3 meses** (2018-05, 2018-06, 2018-07) para backtest.
- Dejar **2018-08** como mes “futuro” para la prueba final del módulo (no tocar para entrenar).

---

## 2. Objetivo del Sprint 2

Construir un **pipeline reproducible y escalable** para el forecast de ingresos mensuales que:

1. Genere una **serie temporal mensual de ingresos** (`monthly_revenue`). - OK
2. Implemente un **split temporal** (train/val, backtest, test final). - 
3. Aplique **limpieza, transformación y creación de features** (≥ 100 features candidatas). - Luis
4. Implemente al menos un **baseline (Naive - ya existe)** y un **primer modelo de regresión**.
5. Aplique alguna técnica de **selección / reducción de variables**. - TODOS
6. Calcule **métricas técnicas** y las traduzca en **KPIs de negocio**. - Felipe 
7. Permita **simular la llegada mensual de nuevos datos** (pipeline corrible cada mes). - Sergio
8. Esté bien documentado (flujo de datos, decisiones y resultados). - TODOS

---

## 3. Dataset y partición temporal

- Fuente: tablas Olist (orders, payments, etc.).
- Target: `monthly_revenue` = suma de `payment_value` por mes.

**Partición propuesta:**

- **Train + Validation**: desde `2016-10` hasta `2018-04`.
- **Backtest**: `2018-05`, `2018-06`, `2018-07`.
- **Test final (no tocar para entrenar)**: `2018-08`.

> La idea es simular el escenario: “hoy” es 2018-08 y queremos ver si lo que aprendimos en los meses anteriores funciona bien en el futuro.

---

## 4. Roles del equipo

- 👷‍♂️ **Ingeniero de Datos (Persona A)**
  - Construye la serie mensual, el pipeline de preprocesamiento y la simulación mensual.
- 🧪 **Científico de Datos (Persona B)**
  - Diseña las features, entrena modelos, mide métricas y aplica selección de variables.
- 🧭 **Data Product Owner – DPO (Persona C)**
  - Define el problema de negocio, KPIs, documenta el flujo y arma el storytelling/slides.

---

## 5. Backlog Sprint 2 – To-Do por rol

### 👷‍♂️ Persona A – Ingeniero de Datos

**A.1 – Serie mensual de ingresos**

- [ ] Cargar los datos raw desde `data/raw` (orders, payments, etc.).
- [ ] Definir la lógica de ingresos mensuales:
  - [ ] Agregar `payment_value` por mes (`year_month`).
  - [ ] Garantizar que el rango es 2016-10 a 2018-08 (verificar mínimo y máximo de fecha).
- [ ] Generar un DataFrame final con, al menos:
  - [ ] `year_month` (ej. 2016-10).
  - [ ] `monthly_revenue`.
  - [ ] Otras variables base que luego usará B para features.

**A.2 – Split temporal reproducible**

- [ ] Implementar una función en `src/preprocess.py` o similar:
  ```python
  def split_by_month(df):
      train_val = df[df["year_month"] <= "2018-04"]
      backtest = df[(df["year_month"] >= "2018-05") & (df["year_month"] <= "2018-07")]
      final_test = df[df["year_month"] == "2018-08"]
      return train_val, backtest, final_test

* [ ] Verificar con prints/logs cuántos meses y filas hay en cada split.

**A.3 – Pipeline de preprocesamiento**

* [ ] Crear módulo(s) para:

  * [ ] Limpieza de nulos y outliers (sin romper la temporalidad).
  * [ ] Transformaciones básicas de columnas (tipos, formateo de fechas, etc.).
  * [ ] Encoding de categóricas (si se usan) y escalado (si el modelo lo requiere).
* [ ] Estructurar archivos:

  * [ ] `src/load_data.py`
  * [ ] `src/preprocess.py`
  * [ ] `src/features.py`
  * [ ] `src/models.py`

**A.4 – Simulación de llegada mensual de datos**

* [ ] Implementar una función tipo:

  ```python
  def run_monthly_pipeline(data_until_month: str):
      """
      Simula que solo existen datos hasta `data_until_month`,
      construye features y entrena/actualiza el modelo para
      predecir el mes siguiente.
      """
      pass
  ```
* [ ] Dejar un ejemplo en el notebook:

  * [ ] “Simular que estamos en 2018-07 y predecir 2018-08”.

**A.5 – Control de versiones y organización**

* [ ] Definir estructura mínima del repo:

  * [ ] `data/raw`, `data/processed`, `data/interim`.
  * [ ] `notebooks/`.
  * [ ] `src/`.
* [ ] Hacer commits con mensajes claros (ej. `feat: add temporal split`).
* [ ] (Opcional) Crear una rama específica de sprint 2.

---

### 🧪 Persona B – Científico de Datos

**B.1 – Feature engineering (≥ 100 features candidatas)**

* [ ] A partir del DataFrame de A, crear funciones en `src/features.py` para:

  * [ ] Features de tiempo:

    * [ ] `month`, `year`, `quarter`.
    * [ ] `month_index` (0,1,2,… para capturar tendencia).
  * [ ] Rezagos de ingresos:

    * [ ] `revenue_lag_1`, `revenue_lag_2`, `revenue_lag_3` (y más si es razonable).
  * [ ] Promedios móviles:

    * [ ] `revenue_ma_3`, `revenue_ma_6` (si hay suficiente historia).
  * [ ] Cambios relativos:

    * [ ] `% cambio` respecto al mes anterior.
  * [ ] Otras features derivadas (por categoría, por región, etc.) que ayuden a llegar a 100+ columnas.

**B.2 – Modelos y baseline**

* [ ] Implementar el **modelo Naive**:

  * [ ] Predicción del mes t = valor observado en t-1.
* [ ] Implementar al menos **un modelo de regresión** sencillo, por ejemplo:

  * [ ] Regresión lineal con `month_index` + lags + MA.
  * [ ] (Opcional) Árbol de decisión, Random Forest, etc.
* [ ] Entrenar modelos usando **train+val** (hasta 2018-04).

**B.3 – Backtest y evaluación**

* [ ] Usar el conjunto de **backtest (2018-05 a 2018-07)** para evaluar:

  * [ ] Naive.
  * [ ] Modelo 1 (regresión).
* [ ] Calcular métricas técnicas:

  * [ ] MAE.
  * [ ] RMSE.
  * [ ] MAPE.
  * [ ] Forecast Bias (promedio de errores).
* [ ] Comparar Naive vs Modelo 1 y dejar conclusiones claras.

**B.4 – Selección de variables**

* [ ] Calcular la **matriz de correlación**:

  * [ ] Identificar variables muy correlacionadas entre sí.
  * [ ] Ver qué features se correlacionan más con `monthly_revenue`.
* [ ] Aplicar al menos una técnica de selección:

  * [ ] Filtrar features por correlación (ej. descartar alta colinealidad).
  * [ ] (Opcional) Usar un modelo tipo Random Forest para importancia de variables.
* [ ] Volver a entrenar el modelo con el subset de features seleccionadas.
* [ ] Comparar métricas:

  * [ ] “Todas las features” vs “features seleccionadas”.
  * [ ] Resumir si la performance se mantiene o mejora con menos variables.

---

### 🧭 Persona C – Data Product Owner (DPO)

**C.1 – Problema de negocio y KPIs**

* [ ] Escribir en el `README` o en el notebook la formulación del problema:

  * [ ] “Predecir los ingresos mensuales del e-commerce para:
    - Optimizar inventario y logística.
    - Planificar estrategias comerciales/marketing.
    - Mejorar la gestión financiera y presupuestaria.”
* [ ] Definir **hipótesis de trabajo** (ejemplo):

  * [ ] “Los ingresos presentan patrones temporales y estacionales que permiten predecirlos con error menor al X%”.
* [ ] Definir **KPIs de negocio**:

  * [ ] Umbral de MAPE aceptable (ej. < 10%).
  * [ ] Interpretar qué significa este error en términos de planificación (breve texto).

**C.2 – Documentación del pipeline y flujo de datos**

* [ ] Crear un diagrama simple (puede ser en PowerPoint, draw.io o markdown) que muestre:

  * [ ] Datos Olist (raw) → limpieza → agregación mensual → features → modelo → métricas.
* [ ] Documentar en el `README`:

  * [ ] Descripción corta de cada etapa del pipeline.
  * [ ] Referencia a los módulos (`load_data.py`, `preprocess.py`, `features.py`, `models.py`).

**C.3 – Diseño del experimento temporal (backtest)**

* [ ] Explicar en texto y/o gráfico:

  * [ ] Qué meses fueron usados para train+val.
  * [ ] Qué meses se reservaron para backtest.
  * [ ] Qué mes se deja como test final (2018-08) y por qué no se usa para entrenar.
* [ ] Definir claramente el concepto de **backtesting** y su importancia.

**C.4 – Slides y storytelling (exposición de 15 minutos)**

* [ ] Diseñar la estructura de la exposición:

  1. **Contexto y problema de negocio** (C).
  2. **Pipeline y manejo temporal** (A).
  3. **Features, modelos y métricas** (B).
* [ ] Asegurar que cada integrante tenga:

  * [ ] 1–2 slides máximo.
  * [ ] Mensajes claros y directos (sin exceso de texto).
* [ ] Verificar que todos los números y gráficos que aparezcan en las slides:

  * [ ] Coinciden con el notebook y las métricas calculadas por B.

**C.5 – Revisión final de consistencia**

* [ ] Confirmar que el mes 2018-08 **no se usó para entrenar ni seleccionar variables**.
* [ ] Revisar que las métricas y conclusiones sean coherentes entre:

  * [ ] Código.
  * [ ] Notebook.
  * [ ] Slides.
* [ ] Dejar una sección final en el README con:

  * [ ] Breve resumen de resultados.
  * [ ] Próximos pasos para Sprint 3 (por ejemplo: probar modelos más avanzados, tuning, etc.).

---

## 6. Entregables del Sprint 2

* [ ] **Notebook principal** con el pipeline completo (carga → features → modelo → métricas).
* [ ] **Scripts modulares** en `src/`:

  * [ ] `load_data.py`, `preprocess.py`, `features.py`, `models.py`, etc.
* [ ] **Documentación del flujo de datos y partición temporal** en este `README` (o `docs/`).
* [ ] **Métricas finales** del pipeline y resumen de KPIs de negocio.
* [ ] **Slides de la exposición** (1–2 por integrante) con storytelling claro.

---

## 7. Cómo correr el pipeline (placeholder)

> Completar cuando A y B terminen los scripts.

Ejemplo:

```bash
# 1. Crear entorno
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Ejecutar notebook o script principal
jupyter notebook notebooks/sprint2_pipeline.ipynb
# o
python src/run_pipeline.py
```

