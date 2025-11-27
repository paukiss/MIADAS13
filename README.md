# Forecast de Ingresos Mensuales — Sprint 3 (Grupo 10)

Este repo plantilla incluye **estructura + código** para el **Producto 3 / Sprint 3**:
- **Hiperparametrización**, validación cruzada y **comparación de modelos**
- Exportación del **modelo final en `.pkl`**
- **Gráficos**: rendimiento (backtest) y feature importance
- Evidencias listas en `reports/` y documentación en `docs/sprint3/`

> Nota: el objetivo es forecast **mensual** (horizonte 1 mes adelante) de `monthly_revenue` usando Olist.

## 1) Setup rápido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Pipeline end-to-end

1) Construir master table mensual (desde URLs o caché):
```bash
python -m src.cli build-master
```

2) Tuning + selección de modelo:
```bash
python -m src.cli tune
```

3) Evaluar (backtest, métricas + plots):
```bash
python -m src.cli evaluate
```

4) Predecir el próximo mes (a partir del último mes disponible):
```bash
python -m src.cli predict
```

## 3) Estructura clave

- `src/forecast_income/` → código (data, features, models)
- `data/processed/master_table_monthly.csv` → master table mensual (se genera)
- `models/final/model.pkl` → artefacto final (se genera en tuning)
- `reports/metrics/metrics_summary.csv` + `reports/figures/*.png` → evidencias Sprint 3
- `docs/sprint3/` → análisis, retraining plan, model card, ética/gobernanza
