# ==============================================================================
# ARQUITECTURA FASE 24: OPTIMIZADOR DE TOLERANCIA (TOLERANCE LOGIC)
# Objetivo: Flexibilizar los filtros para permitir operativa en mercados 
# en recuperación sin comprometer la seguridad de los 1.000 €.
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
st.set_page_config(page_title="Portal IA - Tolerancia", layout="wide", page_icon="⚖️")

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
st.sidebar.header("⚖️ Diales de Tolerancia")

# Dial 1: Probabilidad
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0, 
                                 help="Bajamos ligeramente de 60% a 58% para dar más aire.")

# Dial 2: Margen de Tendencia (LA CLAVE)
margen_tendencia = st.sidebar.slider("Margen de Tendencia (%)", 0.0, 5.0, 1.5, 
                                      help="Permite operar si el precio está hasta un 1.5% por debajo de la Media 50.")

# Dial 3: Volatilidad
vol_min = st.sidebar.slider("Volatilidad Mínima", 0.0, 2.0, 0.5)

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
    
    # RSI (Indicador de agotamiento)
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
# 4. SIMULADOR CON LÓGICA DE TOLERANCIA
# ------------------------------------------------------------------------------
def ejecutar_simulacion_tolerante(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    cap_sim = capital_total
    curva = []
    ops = 0
    rechazos = {"Prob": 0, "Vol": 0, "Tend": 0}
    
    for i in range(len(df) - dias, len(df)):
        estudio = df.iloc[:i]
        hoy = df.iloc[i]
        prob = entrenar_ia(estudio)
        umbral_actual = umbral_base * st.session_state['performance_score']
        
        # --- LÓGICA DE TOLERANCIA ---
        pasa_prob = prob >= umbral_actual
        pasa_vol = hoy['Volatilidad'] > vol_min
        
        # Tolerancia: Precio > (Media 50 - Margen %)
        limite_inferior_tendencia = hoy['Media_50'] * (1 - (margen_tendencia / 100))
        pasa_tend = hoy['Close'] >= limite_inferior_tendencia
        
        if pasa_prob and pasa_vol and pasa_tend:
            ops += 1
            var_futura = (df.iloc[i+1]['Close'] / hoy['Close']) - 1 if i+1 < len(df) else 0
            cap_sim += (cap_sim * 0.2 * var_futura) - (comision_fija * 2)
        else:
            if not pasa_prob: rechazos["Prob"] += 1
            if not pasa_vol: rechazos["Vol"] += 1
            if not pasa_tend: rechazos["Tend"] += 1
            
        curva.append(cap_sim)
        
    return curva, ops, rechazos

# ------------------------------------------------------------------------------
# 5. DASHBOARD
# ------------------------------------------------------------------------------
st.title("⚖️ Portal IA: Optimizador de Tolerancia")

tab1, tab2 = st.tabs(["🚀 Control de Radar", "📊 Test de Tolerancia"])

with tab1:
    st.info(f"Estado del Sistema: Adaptación a {st.session_state['performance_score']:.2f}x")
    if st.sidebar.button("♻️ REINICIAR ADAPTACIÓN"):
        st.session_state['performance_score'] = 1.0
        st.toast("Sistema reseteado")

with tab2:
    st.subheader("Simulación QQQ: Rompiendo la Parálisis")
    if st.button("🏁 Ejecutar Test de Tolerancia"):
        with st.spinner("Ajustando márgenes y recalculando señales..."):
            curva, n_ops, rechazos = ejecutar_simulacion_tolerante("QQQ")
            
            st.line_chart(curva)
            beneficio = curva[-1] - capital_total
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Operaciones", n_ops, delta=f"{n_ops} vs 0 anterior")
            c2.metric("Resultado Final", f"{curva[-1]:.2f} €", delta=f"{beneficio:.2f} €")
            c3.metric("Tapón Tendencia", rechazos["Tend"], delta="Filtro Flexibilizado", delta_color="normal")
            
            if n_ops > 0 and beneficio > -10:
                st.success(f"🎯 ¡CONEXIÓN RECUPERADA! Has logrado realizar {n_ops} operaciones ajustando la tolerancia.")
            elif n_ops == 0:
                st.warning("⚠️ Sigue habiendo parálisis. Prueba a subir el 'Margen de Tendencia' al 3% o bajar la 'Probabilidad' al 55%.")

st.markdown("""
---
**Nota del Arquitecto:** Al permitir un **Margen de Tendencia**, estamos aceptando que el mercado puede estar un poco "sucio" (bajista), pero confiamos en que la IA ha detectado el giro antes que el indicador tradicional. Para 1.000 €, esto es lo que marca la diferencia entre entrar tarde o entrar en el momento justo.
""")
