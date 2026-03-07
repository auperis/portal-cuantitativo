# ==============================================================================
# ARQUITECTURA FASE 23: DETECTOR DE CUELLOS DE BOTELLA (BOTTLENECK DIAGNOSIS)
# Objetivo: Identificar qué filtro está bloqueando las operaciones para ajustar
# la sensibilidad del Sniper y recuperar la rentabilidad.
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
st.set_page_config(page_title="Portal IA - Diagnóstico", layout="wide", page_icon="🔬")

if 'performance_score' not in st.session_state:
    st.session_state['performance_score'] = 1.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES DE INFRAESTRUCTURA
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("💸 Parámetros de Fricción")
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)

# AJUSTES DE SENSIBILIDAD (EL DIAL DEL FRANCOTIRADOR)
st.sidebar.subheader("🎯 Diales del Sniper")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 60.0)
vol_min = st.sidebar.slider("Volatilidad Mínima", 0.0, 2.0, 0.6)
filtro_tendencia = st.sidebar.checkbox("Activar Filtro Tendencia (Media 50)", value=True)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_50'] = d['Close'].rolling(50).mean()
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    # RSI
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
# 4. SIMULADOR CON DIAGNÓSTICO DE FILTROS
# ------------------------------------------------------------------------------
def ejecutar_diagnostico(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    
    cap_sim = capital_total
    curva = []
    rechazos = {"Probabilidad": 0, "Volatilidad": 0, "Tendencia": 0}
    exitos = 0
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        prob = entrenar_ia(estudio)
        umbral_actual = umbral_base * st.session_state['performance_score']
        
        # --- EL ESCÁNER DE RECHAZOS ---
        pasa_prob = prob >= umbral_actual
        pasa_vol = hoy['Volatilidad'] > vol_min
        pasa_tend = (hoy['Close'] > hoy['Media_50']) if filtro_tendencia else True
        
        if pasa_prob and pasa_vol and pasa_tend:
            exitos += 1
            var_futura = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            cap_sim += (cap_sim * 0.2 * var_futura) - (comision_fija * 2)
        else:
            if not pasa_prob: rechazos["Probabilidad"] += 1
            if not pasa_vol: rechazos["Volatilidad"] += 1
            if not pasa_tend: rechazos["Tendencia"] += 1
            
        curva.append(cap_sim)
        
    return curva, exitos, rechazos

# ------------------------------------------------------------------------------
# 5. DASHBOARD DE DIAGNÓSTICO
# ------------------------------------------------------------------------------
st.title("🔬 Portal IA: Diagnóstico de Cuellos de Botella")

tab1, tab2 = st.tabs(["🚀 Control de Vuelo", "📊 Laboratorio de Simulación"])

with tab1:
    umbral_f = umbral_base * st.session_state['performance_score']
    st.info(f"Filtro Sniper: Probabilidad > {umbral_f:.1f}% | Volatilidad > {vol_min}")
    if st.sidebar.button("♻️ REINICIAR ADAPTACIÓN"):
        st.session_state['performance_score'] = 1.0
        st.toast("Sistema reseteado")

with tab2:
    st.subheader("Simulación QQQ: ¿Por qué no operamos?")
    if st.button("🏁 Iniciar Diagnóstico"):
        with st.spinner("Analizando causas de parálisis..."):
            curva, n_ops, logs_rechazo = ejecutar_diagnostico("QQQ")
            
            st.line_chart(curva)
            
            # Visualización de los "Tapones"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Operaciones", n_ops)
            c2.metric("Tapón Probabilidad", logs_rechazo["Probabilidad"], delta="Filtro IA", delta_color="inverse")
            c3.metric("Tapón Volatilidad", logs_rechazo["Volatilidad"], delta="Mercado Lento", delta_color="inverse")
            c4.metric("Tapón Tendencia", logs_rechazo["Tendencia"], delta="Mercado Bajista", delta_color="inverse")
            
            if n_ops == 0:
                st.warning("⚠️ PARÁLISIS DETECTADA: Identifica cuál es el número más alto en los 'Tapones' y ajusta el dial en la barra lateral.")
            else:
                st.success(f"¡Sistema reactivado! Capital final: {curva[-1]:.2f} €")
