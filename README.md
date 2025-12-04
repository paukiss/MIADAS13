# 📊 Forecast de Ingresos - Olist E-Commerce (Grupo 10)

Este proyecto implementa una solución **End-to-End de Machine Learning** para predecir los ingresos mensuales (y diarios) de la plataforma de e-commerce **Olist**. El sistema abarca desde la ingesta de datos crudos hasta el despliegue de un dashboard interactivo para la toma de decisiones.

## 🎯 Objetivo del Proyecto

Desarrollar un modelo predictivo robusto que permita estimar los ingresos futuros de Olist, facilitando la planificación financiera y operativa. El proyecto se centra en:
- **Predicción de Series Temporales**: Utilizando enfoques supervisados (lags, ventanas móviles).
- **Evaluación Rigurosa**: Backtesting con ventana expansiva para simular escenarios reales.
- **Visualización**: Un dashboard interactivo para explorar datos históricos, métricas y predicciones.

## 🚀 Características Principales

- **Pipeline ETL Automatizado**: Procesamiento de múltiples fuentes de datos (pedidos, pagos, clientes) para generar una tabla maestra (`master_table`).
- **Ingeniería de Características**: Generación automática de variables temporales, rezagos (lags) y medias móviles.
- **Modelado Avanzado**: Entrenamiento y optimización de modelos (Gradient Boosting, Random Forest, Ridge) con búsqueda de hiperparámetros.
- **Evaluación de Negocio**: Cálculo de KPIs financieros (Revenue, AOV, ARPU) y métricas de error (MAPE, RMSE).
- **Dashboard Interactivo**: Interfaz web construida con Streamlit para usuarios finales.

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd MIADAS13
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configuración:**
   El archivo `config/params.yaml` controla todos los parámetros del pipeline (rutas de datos, hiperparámetros, ventanas de backtest).

## 💻 Uso del Proyecto

El proyecto se puede ejecutar mediante una interfaz de línea de comandos (CLI) o a través del dashboard.

### 1. Interfaz de Línea de Comandos (CLI)

El pipeline completo se gestiona a través de `src/cli.py`:

- **Construir el Dataset (ETL):**
  Genera la tabla maestra a partir de los datos crudos.
  ```bash
  python -m src.cli build-master
  ```

- **Entrenar y Optimizar (Tuning):**
  Ejecuta la búsqueda de hiperparámetros y guarda el mejor modelo.
  ```bash
  python -m src.cli tune
  ```

- **Evaluar el Modelo (Backtest):**
  Realiza la validación histórica y genera reportes de métricas.
  ```bash
  python -m src.cli evaluate
  ```

- **Generar Predicción:**
  Predice el siguiente periodo (mes/día) utilizando los últimos datos disponibles.
  ```bash
  python -m src.cli predict
  ```

### 2. Dashboard Interactivo (Streamlit)

Para visualizar los resultados y realizar predicciones en tiempo real:

```bash
streamlit run src/app/dashboard.py
```
El dashboard permite:
- Visualizar la serie histórica de ingresos.
- Comparar predicciones vs datos reales (incluyendo escenarios específicos como Agosto 2018).
- Analizar la importancia de variables y métricas detalladas de negocio.

## 📂 Estructura del Proyecto

```
├── config/             # Archivos de configuración (params.yaml)
├── data/               # Datos crudos y procesados
├── docs/               # Documentación del proyecto (Sprints, Análisis)
├── models/             # Modelos entrenados y metadatos
├── notebooks/          # Notebooks de exploración y prototipado
├── reports/            # Reportes generados (Gráficos, Métricas JSON/CSV)
├── src/                # Código fuente
│   ├── app/            # Código del Dashboard (Streamlit)
│   ├── cli.py          # Punto de entrada CLI
│   └── forecast_income/# Paquete principal (Data, Features, Models)
├── tests/              # Tests unitarios
├── README.md           # Documentación general
└── requirements.txt    # Dependencias del proyecto
```

## 📊 Metodología y Resultados

El modelo final seleccionado (generalmente **Gradient Boosting**) ha demostrado un desempeño robusto en las pruebas de backtest.

- **Estrategia de Validación**: Time Series Split / Expanding Window.
- **Métrica Principal**: MAPE (Mean Absolute Percentage Error).
- **Variables Clave**: Ingresos rezagados (Lag 1, 2, 3), Medias móviles (3, 6 meses), Tendencia y Estacionalidad.

Los resultados detallados se pueden consultar en la carpeta `reports/` o directamente en la pestaña "Métricas Detalladas" del dashboard.

---
**Desarrollado por el Grupo 10 - Máster en Inteligencia Artificial y Data Science**
