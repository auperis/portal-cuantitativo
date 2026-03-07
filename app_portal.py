# ==============================================================================
# ARQUITECTURA FASE 17: AJUSTADOR DE VENTAJA NETA (DYNAMIC THRESHOLD)
# Objetivo: Calcular el umbral de convicción mínimo basado en costes y fiscalidad.
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
st.set_page_config(page_title="Portal IA - Ventaja Neta", layout="wide", page_icon="📈")

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES DE INFRAESTRUCTURA
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

# --- AGENTE FISCAL Y COSTES ---
st.sidebar.divider()
st.sidebar.header("⚖️ Parámetros de Fricción")
tasa_impuestos = st.sidebar.slider("Impuestos sobre beneficio (%)", 19, 28, 19)
tipo_activo = st.sidebar.radio("Estrategia Fiscal", ["Acumulación (Eficiente)", "Distribución (Lastre)"])
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)

# --- GESTIÓN DE RIESGO ---
st.sidebar.divider()
st.sidebar.header("🛡️ Gestión de Riesgo")
capital_total = st.sidebar.number_input("Capital en Gestión (€)", value=1000)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)
activar_alertas = st.sidebar.checkbox("Notificar al móvil", value=True)

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
# 5. DASHBOARD: EL AJUSTADOR DE VENTAJA NETA
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Optimizador de Ventaja Neta")
st.markdown(f"### Análisis Institucional de {capital_total} €")

if st.button("🚀 Calcular Ventaja Neta"):
    with st.spinner("Analizando fricción y señales..."):
        data = ejecutar_radar_ia()
        
        if data:
            df_final = pd.DataFrame(data).sort_values("Convicción (%)", ascending=False)
            st.table(df_final)
            
            ganador = df_final.iloc[0]
            precio = ganador["Precio ($)"]
            
            # --- CÁLCULOS DE FRICCION ---
            riesgo_eur = capital_total * 0.02 # 20€
            beneficio_obj_bruto = riesgo_eur * 2 # Objetivo 40€
            
            # El lastre depende del tipo de activo
            if tipo_activo == "Acumulación (Eficiente)":
                lastre_fiscal = 0 # No pagamos hasta vender dentro de años
            else:
                lastre_fiscal = beneficio_obj_bruto * (tasa_impuestos / 100)
            
            costes_totales = (comision_fija * 2) + (beneficio_obj_bruto * 0.005) # Incluye Spread
            friccion_total = lastre_fiscal + costes_totales
            
            # --- CÁLCULO DEL UMBRAL DINÁMICO (LA MAGIA) ---
            # Si la fricción es el 25% del beneficio esperado, necesitamos un 25% más de convicción sobre el 50% base.
            ventaja_necesaria = (friccion_total / beneficio_obj_bruto) * 100
            umbral_dinamico = 50 + ventaja_necesaria
            
            st.divider()
            st.subheader(f"🎯 Diagnóstico Dinámico: {ganador['Activo']}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Convicción IA", f"{ganador['Convicción (%)']}%")
            c2.metric("Umbral de Rentabilidad", f"{umbral_dinamico:.1f}%", help="Mínimo necesario para cubrir gastos.")
            c3.metric("Fricción Total", f"{friccion_total:.2f} €", delta_color="inverse")
            
            # Lógica de Decisión
            if ganador["Convicción (%)"] >= umbral_dinamico:
                st.success(f"✅ VENTAJA NETA POSITIVA: La IA supera el lastre de {friccion_total:.2f} €.")
                
                acciones = riesgo_eur / (precio * (stop_loss_pct / 100))
                reporte = (
                    f"🚀 *ORDEN DE VENTAJA NETA*\n\n"
                    f"Activo: `{ganador['Activo']}`\n"
                    f"IA: `{ganador['Convicción (%)']}%` vs Mínimo: `{umbral_dinamico:.1f}%` ✅\n"
                    f"Acciones: `{acciones:.4f}`\n"
                    f"Fricción: `{friccion_total:.2f} €`"
                )
                if activar_alertas:
                    enviar_alerta(reporte)
                    st.toast("📲 Alerta enviada.")
            else:
                st.error(f"❌ VENTAJA NETA NEGATIVA: Necesitas un {umbral_dinamico:.1f}% de convicción para que esta operación sea rentable.")
                st.info(f"💡 Consejo del Arquitecto: Para bajar el umbral a {50 + ((costes_totales / beneficio_obj_bruto) * 100):.1f}%, cambia a una estrategia de Acumulación.")
