import json
from pathlib import Path

import typer
import yaml

from forecast_income.data.load_raw import download_raw_datasets, load_raw_data
from forecast_income.data.build_monthly_table import build_master_table_monthly, save_master_table
from forecast_income.data.build_daily_table import build_master_table_daily
from forecast_income.models.tune import tune_and_select
from forecast_income.models.evaluate import run_backtest_and_reports
from forecast_income.models.predict import predict_next_month
from forecast_income.utils.io import ensure_dirs
from forecast_income.utils.config import load_config

app = typer.Typer(add_completion=False)

def load_params():
    return load_config("config/params.yaml")

@app.command()
def download_data():
    """Descarga los CSV de Olist y los guarda en data/raw (cache)."""
    params = load_params()
    cache_dir = params["data"]["cache_dir"]
    ensure_dirs([cache_dir])
    download_raw_datasets(base_url=params["data"]["base_url"], cache_dir=cache_dir)
    typer.echo("✅ Datasets descargados en data/raw")

@app.command()
def build_master():
    """Crea la tabla maestra (diaria o mensual) y la guarda en data/processed."""
    params = load_params()
    data = load_raw_data(base_url=params["data"]["base_url"], cache_dir=params["data"]["cache_dir"])
    
    if params.get("active_mode") == "daily":
        master = build_master_table_daily(data)
    else:
        master = build_master_table_monthly(data)
        
    save_master_table(master, out_path=params["data"]["master_table_path"])
    typer.echo(f"✅ Master table guardada en {params['data']['master_table_path']}")

@app.command()
def tune():
    """Ejecuta hiperparametrización y exporta el mejor modelo a models/final."""
    params = load_params()
    ensure_dirs(["models/final", "reports/metrics", "reports/figures"])
    result = tune_and_select(params_path="config/params.yaml")
    typer.echo("✅ Mejor modelo exportado:")
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))

@app.command()
def evaluate():
    """Corre backtesting + métricas + plots."""
    params = load_params()
    ensure_dirs(["reports/metrics", "reports/figures"])
    out = run_backtest_and_reports(params_path="config/params.yaml")
    typer.echo("✅ Reportes generados:")
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))

@app.command()
def predict():
    """Predice ingresos del próximo mes usando el modelo final."""
    params = load_params()
    pred = predict_next_month(params_path="config/params.yaml")
    typer.echo("✅ Predicción siguiente mes:")
    typer.echo(json.dumps(pred, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()
