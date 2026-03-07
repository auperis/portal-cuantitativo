# ==============================================================================
# ARQUITECTURA FASE 19: EL SIMULADOR DE VUELO (BACKTESTING LOCAL)
# Objetivo: Validar la estrategia en el pasado antes de arriesgar los 1.000 €.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import os

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Simulador", layout="wide", page_icon="🎮")

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
st.sidebar.header("⚖️ Fricción y Riesgo")
tipo_activo = st.sidebar.radio("Estrategia Fiscal", ["Acumulación (Eficiente)", "Distribución (Lastre)"])
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital en Gestión (€)", value=1000)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)

# ------------------------------------------------------------------------------
# 3. MOTOR LÓGICO IA
# ------------------------------------------------------------------------------
def entrenar_y_predecir(df_historico, columnas_pistas):
    df = df_historico.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    
    # Entrenamiento
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[columnas_pistas], df['Target'])
    
    # Predicción última fila
    prob = model.predict_proba(df[columnas_pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. FUNCIÓN DE BACKTESTING (EL SIMULADOR)
# ------------------------------------------------------------------------------
def ejecutar_simulacion(ticker, dias_atras=30):
    df = yf.Ticker(ticker).history(period="2y")
    df['Retorno'] = df['Close'].pct_change() * 100
    df['Media_20'] = df['Close'].rolling(20).mean()
    df['Distancia'] = ((df['Close'] / df['Media_20']) - 1) * 100
    df = df.dropna()
    
    pistas = ['Retorno', 'Distancia']
    capital_simulado = capital_total
    historial_curva = []
    
    # Recorremos los últimos X días para simular
    for i in range(len(df) - dias_atras, len(df)):
        ventana_estudio = df.iloc[:i]
        datos_hoy = df.iloc[i]
        
        # IA predice basándose solo en lo que sabía "ese día"
        prob = entrenar_y_predecir(ventana_estudio, pistas)
        
        # Umbral dinámico (simplificado para backtest)
        umbral = 55.5 * st.session_state['performance_score']
        
        if prob >= umbral:
            # Simulamos operación
            precio_entrada = datos_hoy['Close']
            resultado_real_manana = df.iloc[i+1]['Close'] if i+1 < len(df) else precio_entrada
            
            # Resultado neto de la operación
            var_pct = (resultado_real_manana / precio_entrada) - 1
            ganancia_bruta = (capital_simulado * 0.1) * var_pct # Invertimos 10%
            coste_peaje = comision_fija * 2
            
            capital_simulado += (ganancia_bruta - coste_peaje)
        
        historial_curva.append(capital_simulado)
        
    return historial_curva

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🧠 Portal IA: Simulador de Vuelo")

tab1, tab2 = st.tabs(["🎯 Radar en Vivo", "🎮 Simulador de Estrategia"])

with tab1:
    st.info(f"Estado del Sistema: Adaptación a {st.session_state['performance_score']:.2f}x | Umbral: {(55.5 * st.session_state['performance_score']):.1f}%")
    if st.button("🚀 Escaneo de Mercado"):
        # Lógica de radar estándar (abreviada)
        st.write("Analizando activos principales...")
        # ... (Aquí iría la lógica de la Fase 18)

with tab2:
    st.subheader("Simulación de los últimos 30 días")
    activo_sim = st.selectbox("Selecciona Activo para Simular", ["QQQ", "SPY", "BTC-USD"])
    
    if st.button("🏁 Iniciar Simulación Histórica"):
        with st.spinner("Corriendo simulador..."):
            curva = ejecutar_simulacion(activo_sim)
            
            st.line_chart(curva)
            beneficio_final = curva[-1] - capital_total
            
            c1, c2 = st.columns(2)
            c1.metric("Resultado Final", f"{curva[-1]:.2f} €", delta=f"{beneficion_final:.2f} €")
            c2.metric("Eficiencia del Simulador", "Óptima" if beneficio_final > 0 else "Crítica")

# --- BOTONES DE ADAPTACIÓN ---
st.sidebar.divider()
st.sidebar.subheader("🕹️ Entrenamiento")
if st.sidebar.button("👍 ACIERTO"):
    st.session_state['performance_score'] = max(0.9, st.session_state['performance_score'] - 0.05)
if st.sidebar.button("👎 ERROR"):
    st.session_state['performance_score'] = min(1.3, st.session_state['performance_score'] + 0.05)
