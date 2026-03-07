# ==============================================================================
# ARQUITECTURA FASE 26: OPTIMIZADOR DE EXPOSICIÓN (DYNAMIC EXPOSURE)
# Objetivo: Ajustar el tamaño de la apuesta basándose en la fuerza de la señal.
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
st.set_page_config(page_title="Portal IA - Exposición Dinámica", layout="wide", page_icon="⚖️")

if 'auto_bias' not in st.session_state:
    st.session_state['auto_bias'] = 0.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("⚖️ Diales de Precisión")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0)
max_exposicion = st.sidebar.slider("Exposición Máxima (%)", 5.0, 40.0, 25.0, help="Máximo capital a arriesgar en una sola operación.")

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
# 4. SIMULADOR CON EXPOSICIÓN DINÁMICA
# ------------------------------------------------------------------------------
def ejecutar_simulacion_exposicion(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    cap_sim = capital_total
    curva = []
    ops = 0
    
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        prob = entrenar_ia(estudio)
        
        if prob >= umbral_final and hoy['Volatilidad'] > 0.5:
            ops += 1
            # ESCALADO DINÁMICO: 
            # Si prob es umbral_final -> exp = 5%
            # Si prob es 100% -> exp = max_exposicion
            rango_prob = 100 - umbral_final
            exceso_prob = prob - umbral_final
            factor_esfuerzo = exceso_prob / rango_prob
            exposicion_real = 0.05 + (factor_esfuerzo * (max_exposicion/100 - 0.05))
            
            var_futura = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            beneficio_op = (cap_sim * exposicion_real * var_futura) - (comision_fija * 2)
            cap_sim += beneficio_op
            
        curva.append(cap_sim)
    
    return curva, ops, umbral_final

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("⚖️ Portal IA: Optimizador de Exposición")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Seguridad", f"{umbral_f:.1f}%", f"+{st.session_state['auto_bias']}% IA Bias")
with c2:
    st.metric("Capital en Riesgo", f"{capital_total} €")
with c3:
    if st.button("♻️ Reiniciar Aprendizaje"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button("🏁 Iniciar Simulación con Escalado"):
    with st.spinner("Calculando tamaños de apuesta dinámicos..."):
        curva, n_ops, u_final = ejecutar_simulacion_exposicion("QQQ")
        st.line_chart(curva)
        
        beneficio = curva[-1] - capital_total
        
        # AUTO-CORRECCIÓN ACTUALIZADA
        if beneficio < 0 and n_ops > 2:
            st.session_state['auto_bias'] += 1.5
            st.error(f"Pérdida detectada. La IA se vuelve más conservadora (+1.5% al umbral).")
        elif beneficio > 5:
            st.session_state['auto_bias'] = max(-5.0, st.session_state['auto_bias'] - 0.5)
            st.success(f"Beneficio detectado. La IA confía más en su estrategia (-0.5% al umbral).")

        st.write(f"**Análisis:** Se realizaron {n_ops} operaciones con un beneficio neto de {beneficio:.2f} €.")
