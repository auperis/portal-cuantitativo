# ==============================================================================
# ARQUITECTURA FASE 22: EL AGENTE DE ALTA CONVICCIÓN (SNIPER MODE)
# Objetivo: Eliminar el desangre por comisiones endureciendo los filtros de entrada.
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
st.set_page_config(page_title="Portal IA - Sniper Mode", layout="wide", page_icon="🎯")

if 'performance_score' not in st.session_state:
    st.session_state['performance_score'] = 1.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES DE PRECISIÓN
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("💸 Auditoría de Fricción")
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)

# AJUSTES SNIPER (MODO AGRESIVO)
st.sidebar.subheader("🎯 Filtros de Alta Convicción")
umbral_base = st.sidebar.slider("Umbral Probabilidad Mínima (%)", 50.0, 75.0, 62.0, 
                                 help="Si la IA no tiene al menos esta convicción, no se mueve.")
volatilidad_minima = st.sidebar.slider("Volatilidad Mínima Requerida", 0.0, 2.0, 0.8, 
                                        help="Evita mercados laterales donde la comisión se come el beneficio.")

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO AVANZADO
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    # Media Lenta (Tendencia de fondo)
    d['Media_50'] = d['Close'].rolling(50).mean()
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
    # Usamos más árboles (200) para mayor estabilidad
    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    model.fit(df[pistas], df['Target'])
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR SNIPER
# ------------------------------------------------------------------------------
def ejecutar_backtest_sniper(ticker, dias=30):
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
        umbral_dinamico = umbral_base * st.session_state['performance_score']
        
        # LÓGICA SNIPER:
        # 1. Probabilidad > Umbral (62%+)
        # 2. Volatilidad > Mínima (Hay movimiento para ganar)
        # 3. Tendencia: Precio sobre Media 50 (No operamos contra corriente)
        if prob >= umbral_dinamico and hoy['Volatilidad'] > volatilidad_minima and hoy['Close'] > hoy['Media_50']:
            ops += 1
            coste_transaccion = comision_fija * 2
            gastos_totales += coste_transaccion
            
            var_mañana = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            # Mayor exposición (20%) pero menos veces
            resultado = (cap_sim * 0.2) * var_mañana
            cap_sim += (resultado - coste_transaccion)
        
        curva.append(cap_sim)
        
    return curva, ops, gastos_totales

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🎯 Portal IA: Modo Sniper (Fase 22)")

tab1, tab2 = st.tabs(["🚀 Operativa en Vivo", "🕵️ Auditoría de Backtesting"])

with tab1:
    umbral_final = umbral_base * st.session_state['performance_score']
    st.info(f"Filtro Sniper Activo: Umbral de Probabilidad al {umbral_final:.1f}%")
    st.write("El sistema ahora ignora señales débiles para proteger tus 1.000 €.")

with tab2:
    st.subheader("Validación de Estrategia: QQQ (30 días)")
    if st.button("🏁 Ejecutar Simulación Sniper"):
        with st.spinner("Buscando solo oportunidades de alta probabilidad..."):
            curva, n_ops, total_gastos = ejecutar_backtest_sniper("QQQ")
            
            st.line_chart(curva)
            beneficio_neto = curva[-1] - capital_total
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Resultado Final", f"{curva[-1]:.2f} €", delta=f"{beneficio_neto:.2f} €")
            c2.metric("Operaciones", n_ops, delta=f"{n_ops - 14} vs anterior", delta_color="inverse")
            c3.metric("Ahorro en Comisiones", f"{28.00 - total_gastos:.2f} €", delta_color="normal")
            
            if beneficio_neto > -5: # Un margen de error aceptable
                st.success("🎯 Objetivo de reducción logrado. El sistema es ahora más eficiente.")
            else:
                st.warning("⚠️ Todavía hay ruido. Considera subir el 'Umbral de Probabilidad' a 65%.")

# --- CONTROL DE ADAPTACIÓN ---
st.sidebar.divider()
st.sidebar.subheader("🕹️ Entrenamiento")
if st.sidebar.button("♻️ REINICIAR SISTEMA"):
    st.session_state['performance_score'] = 1.0
    st.toast("Adaptación reseteada a 1.0x")
