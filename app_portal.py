# ==============================================================================
# ARQUITECTURA FASE 20: INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# Objetivo: Añadir RSI y Volatilidad para que la IA supere el umbral de adaptación.
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
st.set_page_config(page_title="Portal IA - Inteligencia Avanzada", layout="wide", page_icon="🧠")

if 'performance_score' not in st.session_state:
    st.session_state['performance_score'] = 1.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("⚖️ Parámetros Globales")
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO AVANZADO (Músculo para la IA)
# ------------------------------------------------------------------------------
def calcular_indicadores_avanzados(df):
    d = df.copy()
    # Pista 1: Retorno (Velocidad)
    d['Retorno'] = d['Close'].pct_change() * 100
    # Pista 2: Distancia a Media (Ubicación)
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Distancia'] = ((d['Close'] / d['Media_20']) - 1) * 100
    # Pista 3: Volatilidad (Nerviosismo del mercado)
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    # Pista 4: RSI (Fuerza Relativa - El Telescopio)
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d['RSI'] = 100 - (100 / (1 + rs))
    
    return d.dropna()

def entrenar_y_predecir_avanzado(df_hist, pistas):
    df = df_hist.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[pistas], df['Target'])
    
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR DE VUELO ACTUALIZADO
# ------------------------------------------------------------------------------
def ejecutar_backtest(ticker, dias=30):
    raw_data = yf.Ticker(ticker).history(period="2y")
    df = calcular_indicadores_avanzados(raw_data)
    
    pistas = ['Retorno', 'Distancia', 'Volatilidad', 'RSI']
    capital_sim = capital_total
    curva = []
    operaciones_realizadas = 0
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        
        prob = entrenar_y_predecir_avanzado(estudio, pistas)
        umbral = 55.5 * st.session_state['performance_score']
        
        if prob >= umbral:
            operaciones_realizadas += 1
            var_mañana = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            resultado = (capital_sim * 0.1) * var_mañana # Inversión del 10%
            capital_sim += (resultado - (comision_fija * 2))
        
        curva.append(capital_sim)
        
    return curva, operaciones_realizadas

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🧠 Portal IA: Ingeniería de Características")

tab1, tab2 = st.tabs(["🎯 Radar Adaptativo", "🎮 Simulador de Precisión"])

with tab1:
    umbral_actual = 55.5 * st.session_state['performance_score']
    st.info(f"Factor de Adaptación: {st.session_state['performance_score']:.2f}x | Umbral Requerido: {umbral_actual:.1f}%")
    st.write("Con los nuevos indicadores (RSI y Volatilidad), la IA tiene más 'argumentos' para superar el umbral.")

with tab2:
    st.subheader("Simulador de Estrategia con Pistas Avanzadas")
    activo = st.selectbox("Activo para validar", ["QQQ", "SPY", "BTC-USD"])
    
    if st.button("🏁 Iniciar Simulación"):
        with st.spinner("Procesando datos con RSI y Volatilidad..."):
            curva_datos, num_ops = ejecutar_backtest(activo)
            
            st.line_chart(curva_datos)
            beneficio = curva_datos[-1] - capital_total
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Resultado Final", f"{curva_datos[-1]:.2f} €", delta=f"{beneficio:.2f} €")
            c2.metric("Operaciones", num_ops)
            c3.metric("Estado", "Activo" if num_ops > 0 else "Paralizado")

# --- CONTROL DE ADAPTACIÓN ---
st.sidebar.divider()
st.sidebar.subheader("🕹️ Entrenamiento")
if st.sidebar.button("👍 ACIERTO"):
    st.session_state['performance_score'] = max(0.9, st.session_state['performance_score'] - 0.05)
if st.sidebar.button("👎 ERROR"):
    st.session_state['performance_score'] = min(1.3, st.session_state['performance_score'] + 0.05)
if st.sidebar.button("♻️ REINICIAR ADAPTACIÓN"):
    st.session_state['performance_score'] = 1.0
