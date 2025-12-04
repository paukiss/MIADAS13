# Plan de Trabajo - Sprint 4: Integración, Despliegue y Gobernanza

**Objetivo:** Integrar el modelo de Forecast de Ingresos Mensuales a una herramienta de negocio (Dashboard), implementar trazabilidad y documentar el impacto ético y de gobernanza.

## Tareas

### 1. Infraestructura y Despliegue (Ingeniero de Datos)
- [ ] **Estructura de la App**: Crear directorio `src/app/`.
- [ ] **Dashboard (Streamlit)**: Desarrollar `src/app/dashboard.py` para visualizar:
    - Datos históricos de ingresos.
    - Predicción del mes siguiente.
    - Métricas de desempeño (MAPE, RMSE).
    - Análisis de importancia de variables.
- [ ] **API (FastAPI)**: (Opcional) Crear `src/app/api.py` para exponer el endpoint de predicción.
- [ ] **Containerización**: Crear `Dockerfile` para empaquetar la solución.
- [ ] **Dependencias**: Actualizar `requirements.txt` con `streamlit`, `fastapi`, `uvicorn`.

### 2. Documentación y Gobernanza (Científico de Datos / DPO)
- [ ] **Informe Ético y de Gobernanza**: Crear `docs/sprint4/informe_etico_gobernanza.md`.
    - Expandir sobre privacidad, sesgos, trazabilidad.
    - Incluir análisis de riesgos y mitigación.
- [ ] **Storytelling Demo Day**: Crear `docs/sprint4/storytelling_demo_day.md`.
    - Guion para la presentación final.
    - Estructura: Problema -> Solución (IA) -> Impacto de Negocio -> Demo.

### 3. Integración de Negocio
- [ ] **Métricas de Negocio**: Asegurar que el dashboard muestre métricas relevantes para el negocio (e.g., desviación del presupuesto, tendencia de crecimiento).

## Entregables
1. MVP Funcional (Dashboard en Streamlit).
2. Dockerfile para despliegue.
3. Informe Ético y de Gobernanza.
4. Guion de Storytelling.
