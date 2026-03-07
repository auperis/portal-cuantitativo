# ==============================================================================
# ARQUITECTURA FASE 16: EL AGENTE FISCAL (TAX OPTIMIZATION)
# Objetivo: Calcular el impacto de impuestos y priorizar activos de acumulación.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Optimización Fiscal", layout="wide", page_icon="⚖️")

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES INSTITUCIONALES
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

# --- AGENTE FISCAL (NUEVO) ---
st.sidebar.divider()
st.sidebar.header("⚖️ Agente Fiscal")
tasa_impuestos = st.sidebar.slider("Tramo Impositivo (%)", 19, 28, 19, help="En España, el primer tramo del ahorro es el 19%.")
tipo_activo = st.sidebar.radio("Tipo de Activo", ["Acumulación (Acc)", "Distribución (Dist)"], 
                                help="Los de acumulación no pagan impuestos hasta que vendes.")

# --- COSTES Y RIESGO ---
st.sidebar.divider()
st.sidebar.header("🛡️ Gestión de Riesgo y Costes")
capital_total = st.sidebar.number_input("Capital en Gestión (€)", value=1000)
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)
umbral_conviccion = st.sidebar.slider("Umbral Convicción IA (%)", 50, 85, 70)
activar_alertas = st.sidebar.checkbox("Activar Alertas al Móvil", value=True)

# ------------------------------------------------------------------------------
# 3. MOTOR DE MENSAJERÍA
# ------------------------------------------------------------------------------
def enviar_alerta(mensaje):
    url = f"https://api.telegram.org/bot{token_input}/sendMessage"
    payload = {"chat_id": chat_id_input, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except: pass

# ------------------------------------------------------------------------------
# 4. MOTOR LÓGICO IA
# ------------------------------------------------------------------------------
def ejecutar_radar_ia():
    activos = ["SPY", "QQQ", "BTC-USD", "GLD"]
    resultados = []
    for tick in activos:
        try:
            df = yf.Ticker(tick).history(period="1y")
            df['Retorno'] = df['Close'].pct_change()
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            df = df.dropna()
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df[['Retorno']], df['Target'])
            prob = model.predict_proba(df[['Retorno']].iloc[-1:]) [0][1] * 100
            resultados.append({"Activo": tick, "Convicción (%)": round(prob, 2), "Precio ($)": round(df['Close'].iloc[-1], 2)})
        except: pass
    return resultados

# ------------------------------------------------------------------------------
# 5. DASHBOARD Y AUDITORÍA FISCAL
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Gestión y Eficiencia Fiscal")
st.markdown(f"### Análisis de Cartera de 1.000 €")

if st.button("🚀 Ejecutar Análisis Integral"):
    with st.spinner("Calculando señales y lastre fiscal..."):
        data = ejecutar_radar_ia()
        
        if data:
            df_final = pd.DataFrame(data).sort_values("Convicción (%)", ascending=False)
            st.table(df_final)
            
            ganador = df_final.iloc[0]
            precio = ganador["Precio ($)"]
            
            # --- CÁLCULOS DE EJECUCIÓN ---
            riesgo_eur = capital_total * 0.02
            acciones = riesgo_eur / (precio * (stop_loss_pct / 100))
            inv_bruta = acciones * precio
            coste_peaje = comision_fija + (inv_bruta * 0.001)
            
            # --- CÁLCULO FISCAL ---
            # Imaginamos un beneficio objetivo del doble del riesgo (R:R 1:2)
            beneficio_objetivo_bruto = riesgo_eur * 2
            impuestos_estimados = beneficio_objetivo_bruto * (tasa_impuestos / 100)
            beneficio_neto_real = beneficio_objetivo_bruto - impuestos_estimados - (coste_peaje * 2) # Entrada y salida
            
            st.divider()
            st.subheader(f"🎯 Plan de Vuelo: {ganador['Activo']}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Acciones Fraccionadas", f"{acciones:.4f}")
            c2.metric("Peaje (Entrada)", f"{coste_peaje:.2f} €")
            c3.metric("Lastre Fiscal Estimado", f"{impuestos_estimados:.2f} €", delta="-Hacienda", delta_color="inverse")
            
            # Recomendación Fiscal
            if tipo_activo == "Distribución (Dist)":
                st.warning(f"⚠️ Atención: Al usar activos de Distribución, perderás {impuestos_estimados:.2f} € en cada cobro de dividendos. Considera cambiar a Acumulación para proteger el interés compuesto.")
            else:
                st.info("✅ Eficiencia Óptima: El beneficio se reinvertirá íntegro sin pasar por el fisco hoy.")

            # Validación y Alerta
            if ganador["Convicción (%)"] >= umbral_conviccion:
                st.success("✅ SEÑAL VALIDADA")
                reporte = (
                    f"🚀 *ORDEN IA + FISCAL*\n\n"
                    f"Activo: `{ganador['Activo']}`\n"
                    f"Convicción: `{ganador['Convicción (%)']}%` 📈\n"
                    f"Acciones: `{acciones:.4f}`\n"
                    f"Lastre Fiscal: `{impuestos_estimados:.2f} €` ⚖️\n"
                    f"Neto Esperado: `{beneficio_neto_real:.2f} €`"
                )
                if activar_alertas:
                    enviar_alerta(reporte)
                    st.toast("📲 Alerta fiscal enviada.")
            else:
                st.error("Convicción insuficiente para cubrir costes y lastre.")
