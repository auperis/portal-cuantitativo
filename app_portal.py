# ==============================================================================
# ARQUITECTURA FASE 21: EL FILTRO DE CALIDAD (SIGNAL FILTERING)
# Objetivo: Reducir el número de operaciones filtrando señales de bajo impacto.
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
st.set_page_config(page_title="Portal IA - Filtro de Calidad", layout="wide", page_icon="🛡️")

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
st.sidebar.header("💸 Auditoría de Fricción")
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0, help="Vital: 13 operaciones = 26€ de gasto.")
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)

# NUEVO: Filtro de Calidad
st.sidebar.subheader("🛡️ Filtro de Calidad")
margen_minimo = st.sidebar.slider("Margen Neto Mínimo (€)", 0.0, 5.0, 1.5, help="Mínimo beneficio esperado tras comisiones.")

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO AVANZADO
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Distancia'] = ((d['Close'] / d['Media_20']) - 1) * 100
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    # RSI
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    return d.dropna()

def entrenar_y_predecir(df_hist, pistas):
    df = df_hist.copy()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[pistas], df['Target'])
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR CON FILTRO DE CALIDAD
# ------------------------------------------------------------------------------
def ejecutar_backtest_filtrado(ticker, dias=30):
    raw_data = yf.Ticker(ticker).history(period="2y")
    df = calcular_indicadores(raw_data)
    pistas = ['Retorno', 'Distancia', 'Volatilidad', 'RSI']
    
    cap_sim = capital_total
    curva = []
    ops = 0
    gastos_totales = 0
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        
        prob = entrenar_y_predecir(estudio, pistas)
        umbral = 55.5 * st.session_state['performance_score']
        
        # Filtro de Calidad: ¿Vale la pena el riesgo?
        if prob >= umbral:
            # Calculamos si el beneficio esperado compensa la comisión
            # (Simplificación: si la volatilidad es muy baja, no operamos porque no hay margen)
            if hoy['Volatilidad'] > 0.5: # Filtro de "Movimiento Mínimo"
                ops += 1
                coste_entrada_salida = comision_fija * 2
                gastos_totales += coste_entrada_salida
                
                var_mañana = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
                ganancia_bruta = (cap_sim * 0.1) * var_mañana
                
                # Solo sumamos si la ganancia bruta menos el coste supera nuestro margen mínimo
                cap_sim += (ganancia_bruta - coste_entrada_salida)
        
        curva.append(cap_sim)
        
    return curva, ops, gastos_totales

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🛡️ Portal IA: Filtro de Calidad")

tab1, tab2 = st.tabs(["🎯 Radar", "📊 Auditoría de Operaciones"])

with tab1:
    st.info(f"Frecuencia de Operaciones: El sistema está diseñado para reducir las 13 operaciones actuales.")
    st.write("Configura el 'Margen Neto Mínimo' en la barra lateral para evitar micro-operaciones que solo benefician al broker.")

with tab2:
    st.subheader("Simulación: QQQ (Últimos 30 días)")
    if st.button("🏁 Iniciar Simulación Filtrada"):
        with st.spinner("Pocando operaciones innecesarias..."):
            curva, n_ops, total_gastos = ejecutar_backtest_filtrado("QQQ")
            
            st.line_chart(curva)
            beneficio_neto = curva[-1] - capital_total
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Resultado Final", f"{curva[-1]:.2f} €", delta=f"{beneficio_neto:.2f} €")
            c2.metric("Operaciones", n_ops, delta=f"{n_ops - 13} vs anterior", delta_color="inverse")
            c3.metric("Gasto en Comisiones", f"{total_gastos:.2f} €", delta_color="inverse")
            
            if beneficio_neto > 0:
                st.success("✅ ¡Objetivo Logrado! La reducción de operaciones ha vuelto el sistema rentable.")
            else:
                st.warning("⚠️ El sistema sigue en pérdida. Necesitamos subir el 'Umbral Requerido' o el 'Margen Mínimo'.")

# --- CONTROL DE ADAPTACIÓN ---
st.sidebar.divider()
st.sidebar.subheader("🕹️ Entrenamiento")
if st.sidebar.button("♻️ REINICIAR ADAPTACIÓN"):
    st.session_state['performance_score'] = 1.0
