# ==============================================================================
# ARQUITECTURA FASE 25.1: AGENTE DE AUTO-CORRECCIÓN (CORRECCIÓN DE NOMBRE)
# Objetivo: Corregir el NameError al llamar a la función de simulación.
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
st.set_page_config(page_title="Portal IA - Auto-Corrección", layout="wide", page_icon="🤖")

# Memoria de rendimiento autónomo
if 'auto_bias' not in st.session_state:
    st.session_state['auto_bias'] = 0.0 # Ajuste automático del umbral

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("⚖️ Diales de Tolerancia")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0)
margen_tendencia = st.sidebar.slider("Margen de Tendencia (%)", 0.0, 5.0, 1.5)

st.sidebar.divider()
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_50'] = d['Close'].rolling(50).mean()
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    return d.dropna()

def entrenar_ia(df_hist):
    df = df_hist.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[pistas], df['Target'])
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR CON AUTO-CORRECCIÓN
# ------------------------------------------------------------------------------
def ejecutar_simulacion_autonoma(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    cap_sim = capital_total
    curva = []
    ops = 0
    
    # El umbral ahora incluye el "Sesgo Automático" del sistema
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        prob = entrenar_ia(estudio)
        
        pasa_prob = prob >= umbral_final
        limite_inf = hoy['Media_50'] * (1 - (margen_tendencia / 100))
        pasa_tend = hoy['Close'] >= limite_inf
        
        if pasa_prob and pasa_tend and hoy['Volatilidad'] > 0.5:
            ops += 1
            var_futura = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            cap_sim += (cap_sim * 0.2 * var_futura) - (comision_fija * 2)
            
        curva.append(cap_sim)
    
    return curva, ops, umbral_final

# ------------------------------------------------------------------------------
# 5. DASHBOARD
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Agente de Auto-Corrección")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Simulación y Aprendizaje")
    # CORRECCIÓN: Se cambió 'ejecutar_autonoma' por 'ejecutar_simulacion_autonoma'
    if st.button("🏁 Ejecutar Simulación y Auto-Corregir"):
        with st.spinner("Simulando y ajustando redes neuronales..."):
            curva, n_ops, u_final = ejecutar_simulacion_autonoma("QQQ")
            st.line_chart(curva)
            
            beneficio = curva[-1] - capital_total
            
            # --- LÓGICA DE AUTO-CORRECCIÓN ---
            if beneficio < 0 and n_ops > 3:
                # Si perdemos dinero operando mucho, subimos la exigencia
                st.session_state['auto_bias'] += 2.0
                st.warning(f"⚠️ Rendimiento negativo. El Agente ha subido el umbral a {u_final + 2.0:.1f}% para filtrar el ruido.")
            elif beneficio > 10:
                # Si ganamos bien, podemos permitirnos ser un poco más flexibles
                st.session_state['auto_bias'] -= 1.0
                st.success(f"✅ Rendimiento positivo. El Agente ha bajado el umbral a {u_final - 1.0:.1f}% para capturar más señales.")

with col2:
    st.metric("Sesgo de Auto-Corrección", f"+{st.session_state['auto_bias']}%")
    st.info(f"Este valor se suma a tu umbral base para protegerte del overtrading.")
    if st.button("♻️ Resetear IA"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

# Espacio para el Radar real
st.divider()
st.subheader("📡 Radar de Ejecución en Vivo")
st.write("Pulsa el botón de simulación primero para que la IA se ajuste al mercado actual.")
