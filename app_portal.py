# ==============================================================================
# ARQUITECTURA FASE 15: EL AUDITOR DE COSTES (Slippage & Commissions)
# Objetivo: Descontar costes de transacción antes de validar señales de la IA.
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
st.set_page_config(page_title="Portal IA - Auditoría de Costes", layout="wide", page_icon="💸")

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES DE INFRAESTRUCTURA
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

# --- AUDITOR DE COSTES (NUEVO) ---
st.sidebar.divider()
st.sidebar.header("💸 Auditor de Costes")
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0, step=0.5, help="Coste fijo por cada operación.")
slippage_esperado = st.sidebar.slider("Slippage/Deslizamiento (%)", 0.0, 0.5, 0.1, 0.05, help="Diferencia de precio al ejecutar.")

# --- GESTIÓN DE RIESGO ---
st.sidebar.divider()
st.sidebar.header("🛡️ Gestión de Riesgo")
capital_total = st.sidebar.number_input("Capital en Gestión (€)", value=1000)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)
umbral_conviccion = st.sidebar.slider("Umbral Convicción IA (%)", 50, 85, 68)
activar_alertas = st.sidebar.checkbox("Activar Alertas al Móvil", value=True)

# ------------------------------------------------------------------------------
# 3. MOTOR DE MENSAJERÍA
# ------------------------------------------------------------------------------
def enviar_alerta(mensaje):
    url = f"https://api.telegram.org/bot{token_input}/sendMessage"
    payload = {"chat_id": chat_id_input, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except:
        pass

# ------------------------------------------------------------------------------
# 4. MOTOR LÓGICO IA + AUDITORÍA DE COSTES
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
            
            # IA de Convicción
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df[['Retorno']], df['Target'])
            
            prob = model.predict_proba(df[['Retorno']].iloc[-1:]) [0][1] * 100
            precio_actual = df['Close'].iloc[-1]
            
            resultados.append({
                "Activo": tick,
                "Convicción (%)": round(prob, 2),
                "Precio ($)": round(precio_actual, 2)
            })
        except: pass
    return resultados

# ------------------------------------------------------------------------------
# 5. DASHBOARD Y EJECUCIÓN
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Inteligencia Cuantitativa")
st.markdown(f"### Gestión Institucional de 1.000 €")

if st.button("🚀 Iniciar Escaneo y Auditoría de Costes"):
    with st.spinner("Calculando convergencias y peajes de mercado..."):
        data = ejecutar_radar_ia()
        
        if data:
            df_final = pd.DataFrame(data).sort_values("Convicción (%)", ascending=False)
            st.subheader("📊 Ranking de Probabilidades")
            st.table(df_final)
            
            ganador = df_final.iloc[0]
            
            # --- CÁLCULO DE AUDITORÍA DE COSTES ---
            precio = ganador["Precio ($)"]
            riesgo_eur = capital_total * 0.02 # Arriesgamos 20€ de los 1.000€
            perdidia_por_accion = precio * (stop_loss_pct / 100)
            acciones_a_comprar = riesgo_eur / perdidia_por_accion
            
            inversion_bruta = acciones_a_comprar * precio
            coste_peaje = comision_fija + (inversion_bruta * (slippage_esperado / 100))
            
            st.divider()
            st.subheader(f"🎯 Análisis de Ejecución: {ganador['Activo']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Acciones (Fraccionadas)", f"{acciones_a_comprar:.4f}")
            col2.metric("Inversión Bruta", f"{inversion_bruta:.2f} €")
            col3.metric("Coste del Peaje", f"{coste_peaje:.2f} €", delta="-Comisión", delta_color="inverse")
            
            # Validación Final del Auditor
            if ganador["Convicción (%)"] >= umbral_conviccion:
                st.success(f"✅ SEÑAL VALIDADA: La convicción del {ganador['Convicción (%)']}% compensa los costes.")
                
                reporte_telegram = (
                    f"🚀 *ORDEN DE COMPRA IA*\n\n"
                    f"Activo: `{ganador['Activo']}`\n"
                    f"Convicción: `{ganador['Convicción (%)']}%` 📈\n"
                    f"Precio: `{precio} $`\n\n"
                    f"📦 *Instrucción:* Comprar `{acciones_a_comprar:.4f}` acciones.\n"
                    f"💸 *Coste Peaje:* `{coste_peaje:.2f} €`"
                )
                
                if activar_alertas:
                    enviar_alerta(reporte_telegram)
                    st.toast("📲 Alerta de ejecución enviada a Telegram.")
            else:
                st.error(f"❌ SEÑAL RECHAZADA: Convicción insuficiente ({ganador['Convicción (%)']}%).")
