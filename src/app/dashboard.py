import streamlit as st
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
import os
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from forecast_income.models.predict import predict_next_month
from forecast_income.utils.config import load_config

st.set_page_config(page_title="Forecast de Ingresos - Olist", layout="wide")

st.title("📊 Forecast de Ingresos Mensuales - Olist")
st.markdown("### Sprint 4: Integración y Despliegue")

# Load Config
try:
    params = load_config("config/params.yaml")
except Exception as e:
    st.error(f"Error cargando configuración: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configuración")
if st.sidebar.button("🔄 Recargar Datos"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"**Proyecto:** {params['project']['name']}")
st.sidebar.markdown(f"**Dataset:** {params['project']['dataset']}")
st.sidebar.markdown(f"**Horizonte:** {params['project']['horizon']} mes(es)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Parámetros del Modelo:**")
st.sidebar.markdown(f"- **Splits CV:** {params['modeling']['cv_splits']}")
st.sidebar.markdown(f"- **Métrica:** {params['modeling']['primary_metric']}")

# Load Metadata
metadata_path = Path("models/final/metadata.json")
if metadata_path.exists():
    try:
        meta = json.loads(metadata_path.read_text())
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏆 Mejor Modelo Actual")
        st.sidebar.info(f"**{meta.get('best_model', 'Unknown').upper()}**")
        st.sidebar.markdown(f"**MAPE Entrenamiento:** {meta.get('best_mape', 0):.2f}%")
        st.sidebar.caption(f"Actualizado: {meta.get('trained_at_utc', '')[:10]}")
    except:
        pass

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(params["data"]["master_table_path"])
        df[params["data"]["index_col"]] = pd.to_datetime(df[params["data"]["index_col"]])
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(["🔮 Predicción & Validación", "📊 Análisis del Modelo", "📈 Métricas Detalladas"])

    # --- TAB 1: Predicción & Validación ---
    with tab1:
        col1, col2 = st.columns([1, 1])

        # Initialize session state for prediction
        if 'prediction_result' not in st.session_state:
            st.session_state['prediction_result'] = None

        with col1:
            st.subheader("📈 Histórico de Ingresos")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Determine revenue column based on available columns
            if "daily_revenue" in df.columns:
                y_col = "daily_revenue"
                title = "Ingresos Diarios"
            elif "monthly_revenue" in df.columns:
                y_col = "monthly_revenue"
                title = "Ingresos Mensuales"
            else:
                y_col = params["features"]["target_col"]
                title = "Ingresos"

            # --- Visualización Segmentada (Train, Backtest, Target) ---
            date_col = params["data"]["index_col"]
            
            # Configuración de fechas
            target_date = pd.to_datetime("2018-08-01")
            test_size = params.get("modes", {}).get("monthly", {}).get("modeling", {}).get("test_size", 3)
            
            # Definir límites
            backtest_end_date = target_date - pd.DateOffset(months=1) # July
            backtest_start_date = backtest_end_date - pd.DateOffset(months=test_size - 1) # May
            train_end_date = backtest_start_date - pd.DateOffset(months=1) # April

            # Crear máscaras con superposición para continuidad visual
            train_mask = df[date_col] <= train_end_date
            # Backtest conecta desde el último punto de train hasta el último de backtest
            backtest_mask = (df[date_col] >= train_end_date) & (df[date_col] <= backtest_end_date)
            # Target conecta desde el último punto de backtest hasta el target
            target_mask = (df[date_col] >= backtest_end_date) & (df[date_col] <= target_date)

            # Plotear segmentos
            sns.lineplot(data=df[train_mask], x=date_col, y=y_col, ax=ax, marker="o", color="royalblue", label="Train (Histórico)")
            sns.lineplot(data=df[backtest_mask], x=date_col, y=y_col, ax=ax, marker="o", color="orange", label="Backtest (Validación)")
            
            # Solo plotear segmento target si existe el dato de agosto
            if df[date_col].max() >= target_date:
                sns.lineplot(data=df[target_mask], x=date_col, y=y_col, ax=ax, marker="o", color="green", label="Real (Agosto)")
            
            # Overlay Prediction if available
            if st.session_state['prediction_result']:
                result = st.session_state['prediction_result']
                if "prediction_monthly_revenue" in result:
                    pred_val = result["prediction_monthly_revenue"]
                    # Assuming prediction is for 2018-08-01
                    pred_date = pd.to_datetime("2018-08-01")
                    ax.scatter([pred_date], [pred_val], color='red', s=100, zorder=5, label="Predicción Modelo")
                    # Add vertical line from point to x-axis
                    ax.vlines(x=pred_date, ymin=0, ymax=pred_val, colors='red', linestyles='dashed', alpha=0.5)
                    
                elif "daily_predictions" in result:
                    pred_df = pd.DataFrame(result["daily_predictions"])
                    pred_df["date"] = pd.to_datetime(pred_df["date"])
                    sns.lineplot(data=pred_df, x="date", y="daily_revenue", ax=ax, color='red', linestyle='--', label="Predicción Modelo")

            ax.set_title(title)
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Ingresos (BRL)")
            
            # Format X-axis to show all months
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=90)
            
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("🔮 Predicción Próximo Mes")
            
            if st.button("Generar Predicción"):
                with st.spinner("Calculando predicción ..."):
                    try:
                        # Predict starting from July 31st
                        result = predict_next_month("config/params.yaml", cutoff_date="2018-07-31")
                        st.session_state['prediction_result'] = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error en la predicción: {e}")

            if st.session_state['prediction_result']:
                result = st.session_state['prediction_result']
                st.success("Predicción Exitosa")
                
                if "daily_predictions" in result:
                    pred_df = pd.DataFrame(result["daily_predictions"])
                    pred_df["date"] = pd.to_datetime(pred_df["date"])
                    
                    # Get Real August Data
                    august_mask = (df[params["data"]["index_col"]].dt.year == 2018) & (df[params["data"]["index_col"]].dt.month == 8)
                    real_august = df[august_mask].copy()
                    real_august = real_august.rename(columns={params["data"]["index_col"]: "date"})
                    
                    # Merge for comparison
                    comparison = pd.merge(real_august[["date", "daily_revenue"]], pred_df, on="date", how="outer", suffixes=("_real", "_pred"))
                    
                    # Metrics
                    total_real = comparison["daily_revenue_real"].sum()
                    total_pred = comparison["daily_revenue_pred"].sum()
                    diff = total_pred - total_real
                    mape_aug = (abs(comparison["daily_revenue_real"] - comparison["daily_revenue_pred"]) / comparison["daily_revenue_real"]).mean() * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Ingresos Reales (Ago)", f"R$ {total_real:,.2f}")
                    col_m2.metric("Ingresos Predichos (Ago)", f"R$ {total_pred:,.2f}", delta=f"{diff:,.2f}")
                    col_m3.metric("MAPE Agosto", f"{mape_aug:.2f}%")
                    
                    st.markdown("#### 🆚 Comparativa Real vs Predicción (Agosto 2018)")
                    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
                    
                    sns.lineplot(data=comparison, x="date", y="daily_revenue_real", ax=ax_pred, label="Real", marker="o", color="blue")
                    sns.lineplot(data=comparison, x="date", y="daily_revenue_pred", ax=ax_pred, label="Predicción", marker="x", color="orange", linestyle="--")
                    
                    ax_pred.set_title("Agosto 2018: Realidad vs Modelo")
                    ax_pred.set_ylabel("Ingresos (BRL)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig_pred)
                    
                    st.markdown("##### Detalle Diario")
                    st.dataframe(comparison.style.format({
                        "daily_revenue_real": "R$ {:,.2f}",
                        "daily_revenue_pred": "R$ {:,.2f}"
                    }))

                elif "prediction_monthly_revenue" in result:
                    # Monthly Prediction Logic
                    pred_value = result["prediction_monthly_revenue"]
                    
                    # Get Real August Data
                    # Check if we are in monthly mode (index_col is month)
                    date_col = params["data"]["index_col"]
                    august_mask = (df[date_col].dt.year == 2018) & (df[date_col].dt.month == 8)
                    
                    real_august_value = 0
                    if "monthly_revenue" in df.columns:
                        real_august_value = df[august_mask]["monthly_revenue"].sum()
                    elif "daily_revenue" in df.columns:
                        real_august_value = df[august_mask]["daily_revenue"].sum()
                    
                    diff = pred_value - real_august_value
                    mape_aug = 0
                    if real_august_value > 0:
                        mape_aug = (abs(real_august_value - pred_value) / real_august_value) * 100
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Ingresos Reales (Ago)", f"R$ {real_august_value:,.2f}")
                    col_m2.metric("Ingresos Predichos (Ago)", f"R$ {pred_value:,.2f}", delta=f"{diff:,.2f}")
                    col_m3.metric("MAPE Agosto", f"{mape_aug:.2f}%")
                    
                    st.markdown("#### 🆚 Comparativa Real vs Predicción (Agosto 2018)")
                    
                    # Create a small dataframe for plotting
                    comp_df = pd.DataFrame({
                        "Tipo": ["Real", "Predicción"],
                        "Ingresos": [real_august_value, pred_value]
                    })
                    
                    fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
                    sns.barplot(data=comp_df, x="Tipo", y="Ingresos", ax=ax_comp, palette=["blue", "orange"], hue="Tipo", legend=False)
                    ax_comp.set_title("Agosto 2018: Realidad vs Modelo")
                    ax_comp.set_ylabel("Ingresos (BRL)")
                    
                    # Add value labels on bars
                    for i, v in enumerate(comp_df["Ingresos"]):
                        ax_comp.text(i, v, f"R$ {v:,.0f}", ha='center', va='bottom')
                        
                    st.pyplot(fig_comp)

    # --- TAB 2: Análisis del Modelo ---
    with tab2:
        st.header("📊 Análisis del Modelo")
        
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.subheader("Importancia de Variables (Pipeline)")
            if Path("reports/figures/feature_importance.png").exists():
                st.image("reports/figures/feature_importance.png", caption="Feature Importance (Generado en Entrenamiento)")
            else:
                st.info("Imagen de Feature Importance no encontrada.")

        with col_img2:
            st.subheader("Matriz de Correlación")
            if Path("reports/figures/correlation_matrix.png").exists():
                st.image("reports/figures/correlation_matrix.png", caption="Correlación de Variables Seleccionadas")
            else:
                st.info("Imagen de Matriz de Correlación no encontrada.")

        st.markdown("---")
        st.subheader("📋 Variables Seleccionadas")
        selected_features_path = Path("models/final/selected_features.json")
        if selected_features_path.exists():
            try:
                feats = json.loads(selected_features_path.read_text())
                st.write(f"El modelo utiliza **{len(feats)}** variables predictoras:")
                
                # Display as a nice list or dataframe
                feat_df = pd.DataFrame(feats, columns=["Variable"])
                st.dataframe(feat_df, use_container_width=True)
            except:
                st.error("Error leyendo selected_features.json")

    # --- TAB 3: Métricas Detalladas ---
    with tab3:
        st.header("📈 Métricas Detalladas")

        # 1. Backtest Metrics JSON (KPIs)
        metrics_json_path = Path("reports/metrics/backtest_metrics.json")
        if metrics_json_path.exists():
            try:
                bt_metrics = json.loads(metrics_json_path.read_text())
                
                st.subheader("KPIs de Negocio (Backtest)")
                
                # Fila 1: Ingresos y Realización
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Ingresos Totales (Real)", f"R$ {bt_metrics.get('Total Revenue (Real)', 0):,.2f}")
                kpi2.metric("Pronóstico Total (Pred)", f"R$ {bt_metrics.get('Total Forecast (Pred)', 0):,.2f}")
                kpi3.metric("Tasa de Realización", f"{bt_metrics.get('Forecast Realization Rate (%)', 0):.2f}%")
                kpi4.metric("Desviación de Ingresos", f"R$ {bt_metrics.get('Revenue Deviation (R$)', 0):,.2f}")
                
                st.markdown("---")
                
                # Fila 2: Métricas de Negocio (AOV, ARPU, Growth)
                kpi5, kpi6, kpi7, kpi8 = st.columns(4)
                kpi5.metric("Ticket Promedio (AOV)", f"R$ {bt_metrics.get('Avg Ticket (AOV)', 0):,.2f}")
                kpi6.metric("ARPU Promedio", f"R$ {bt_metrics.get('Avg ARPU', 0):,.2f}")
                kpi7.metric("Crecimiento Mensual Promedio", f"{bt_metrics.get('Avg Growth MoM (%)', 0):.2f}%")
                kpi8.metric("Ratio de Flete Promedio", f"{bt_metrics.get('Avg Freight Ratio (%)', 0):.2f}%")
                
                st.markdown("---")
                
                # Fila 3: Operacional (Orders, Sellers, Category)
                kpi9, kpi10, kpi11 = st.columns(3)
                kpi9.metric("Pedidos Promedio/Cliente", f"{bt_metrics.get('Avg Orders/Customer', 0):.2f}")
                kpi10.metric("Total Nuevos Vendedores", f"{bt_metrics.get('Total New Sellers', 0)}")
                kpi11.metric("Categoría Principal", f"{bt_metrics.get('Top Category (Most Freq)', 'N/A')}")

                with st.expander("Ver todas las métricas calculadas (Crudo)"):
                    # Convert dict to dataframe for nice display
                    metrics_list = [{"Métrica": k, "Valor": v} for k, v in bt_metrics.items()]
                    metrics_df_display = pd.DataFrame(metrics_list)
                    # Force string conversion to avoid PyArrow mixed-type errors
                    metrics_df_display["Valor"] = metrics_df_display["Valor"].astype(str)
                    st.dataframe(metrics_df_display)
            except Exception as e:
                st.error(f"Error leyendo backtest_metrics.json: {e}")

        st.markdown("---")
        st.subheader("🏆 Comparativa de Modelos Candidatos")

        metrics_path = Path("reports/metrics/metrics_summary.csv")
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            
            col_metrics1, col_metrics2 = st.columns([1, 1])
            
            with col_metrics1:
                st.markdown("##### Tabla de Métricas")
                st.dataframe(metrics_df[["model", "MAE", "RMSE", "MAPE"]].style.highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE"], color="lightgreen"))
            
            with col_metrics2:
                st.markdown("##### Comparación de Modelos (MAPE)")
                fig_metrics, ax_metrics = plt.subplots(figsize=(6, 4))
                sns.barplot(data=metrics_df, x="model", y="MAPE", ax=ax_metrics, palette="viridis", hue="model", legend=False)
                ax_metrics.set_title("MAPE por Modelo")
                ax_metrics.set_ylabel("MAPE (%)")
                ax_metrics.set_xlabel("Modelo")
                plt.xticks(rotation=45)
                st.pyplot(fig_metrics)
            
        else:
            st.warning("No se encontraron métricas de backtest. Ejecute el pipeline de evaluación.")

        # Backtest Predictions Chart
        backtest_path = Path("reports/metrics/backtest_predictions.csv")
        if backtest_path.exists():
            st.markdown("---")
            st.subheader("📉 Desempeño Histórico (Backtest)")
            bt_df = pd.read_csv(backtest_path)
            if "date" in bt_df.columns:
                bt_df["date"] = pd.to_datetime(bt_df["date"])
                
                fig_bt, ax_bt = plt.subplots(figsize=(12, 4))
                sns.lineplot(data=bt_df, x="date", y="y_true", label="Real", ax=ax_bt, marker="o", color="blue")
                sns.lineplot(data=bt_df, x="date", y="y_pred", label="Predicción", ax=ax_bt, marker="x", linestyle="--", color="orange")
                ax_bt.set_title("Backtest: Real vs Predicción")
                ax_bt.set_ylabel("Ingresos (BRL)")
                ax_bt.set_xlabel("Fecha")
                st.pyplot(fig_bt)
            else:
                st.warning("El archivo de predicciones de backtest no tiene columna de fecha.")
