# ==============================================================================
# ARQUITECTURA FASE 28: EL PARACAÍDAS INTELIGENTE (ATR TRAILING STOP)
# Objetivo: Adaptar la distancia del Stop-Loss a la volatilidad real del mercado
# para evitar salidas prematuras por el "ruido" diario (Whipsaw).
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
st.set_page_config(page_title="Portal IA - Paracaídas ATR", layout="wide", page_icon="🪂")

if 'auto_bias' not in st.session_state:
    st.session_state['auto_bias'] = 0.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN Y RIESGO
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)

st.sidebar.divider()
st.sidebar.header("⚖️ Diales de Precisión")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0)
max_exposicion = st.sidebar.slider("Exposición Máxima (%)", 5.0, 40.0, 25.0)

st.sidebar.divider()
st.sidebar.header("🪂 Sensor de Turbulencias (ATR)")
# SUSTITUIMOS EL PORCENTAJE FIJO POR UN MULTIPLICADOR DE VOLATILIDAD
multiplicador_atr = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0, 
                                      help="2.0x significa que damos de margen el doble de lo que el activo se mueve en un día normal.")

st.sidebar.divider()
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO (AÑADIENDO EL SENSOR ATR)
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_50'] = d['Close'].rolling(50).mean()
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    # RSI
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # NUEVO: CÁLCULO DEL ATR (Average True Range)
    d['High_Low'] = d['High'] - d['Low']
    d['High_PrevClose'] = np.abs(d['High'] - d['Close'].shift(1))
    d['Low_PrevClose'] = np.abs(d['Low'] - d['Close'].shift(1))
    # El True Range es el movimiento máximo del día
    d['True_Range'] = d[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    # Suavizamos la media a 14 días
    d['ATR'] = d['True_Range'].rolling(14).mean()
    # Lo convertimos a porcentaje respecto al precio actual para usarlo de Stop
    d['ATR_pct'] = (d['ATR'] / d['Close']) * 100
    
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
# 4. SIMULADOR CON ESCUDO ATR DINÁMICO
# ------------------------------------------------------------------------------
def ejecutar_simulacion_atr(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo_alcanzado = 0
    stop_dinamico_actual = 0.0 # Guardará el % exacto en cada momento
    
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    for i in range(len(df) - dias, len(df)):
        hoy = df.iloc[i]
        
        # 1. GESTIONAR EXTRACCIÓN (Si estamos dentro)
        if en_posicion:
            precio_maximo_alcanzado = max(precio_maximo_alcanzado, hoy['Close'])
            
            # EL PARACAÍDAS INTELIGENTE: Calculamos la distancia permitida hoy
            # Ejemplo: Si el ATR es 1.5% y el multiplicador 2.0x, el stop es 3.0%
            margen_permitido_pct = hoy['ATR_pct'] * multiplicador_atr
            precio_stop_loss = precio_maximo_alcanzado * (1 - (margen_permitido_pct / 100))
            
            # Guardamos la métrica para el dashboard
            stop_dinamico_actual = margen_permitido_pct
            
            if hoy['Close'] <= precio_stop_loss or i == len(df) - 1:
                valor_venta = acciones_compradas * hoy['Close']
                liquidez += (valor_venta - comision_fija)
                en_posicion = False
                acciones_compradas = 0
                precio_maximo_alcanzado = 0
                stop_dinamico_actual = 0.0
                
        # 2. BUSCAR ENTRADA (Si estamos fuera)
        if not en_posicion and i < len(df) - 1:
            estudio = df.iloc[:i]
            prob = entrenar_ia(estudio)
            
            if prob >= umbral_final and hoy['Volatilidad'] > 0.5:
                ops += 1
                en_posicion = True
                precio_maximo_alcanzado = hoy['Close']
                
                rango_prob = 100 - umbral_final
                factor = min(max((prob - umbral_final) / rango_prob, 0), 1) if rango_prob > 0 else 0
                exposicion_pct = 0.05 + (factor * (max_exposicion/100 - 0.05))
                
                dinero_a_invertir = liquidez * exposicion_pct
                liquidez -= comision_fija
                
                acciones_compradas = dinero_a_invertir / hoy['Close']
                liquidez -= dinero_a_invertir
                
        # 3. FOTOGRAFÍA DIARIA
        valor_cartera_hoy = liquidez + (acciones_compradas * hoy['Close']) if en_posicion else liquidez
        curva_capital.append(valor_cartera_hoy)
        
    # Extraemos el último ATR calculado para mostrarlo en la UI
    ultimo_atr_pct = df.iloc[-1]['ATR_pct']
        
    return curva_capital, ops, umbral_final, ultimo_atr_pct

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🪂 Portal IA: Escudo ATR Inteligente")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Decisión IA", f"{umbral_f:.1f}%", f"{st.session_state['auto_bias']}% Auto-Corrección")
with c2:
    st.metric("Multiplicador de Turbulencia", f"{multiplicador_atr}x", "Ajuste del ATR")
with c3:
    if st.button("♻️ Reiniciar Memoria IA"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button("🏁 Iniciar Simulación ATR (30 Días)"):
    with st.spinner("Midiendo turbulencias del mercado y ajustando paracaídas..."):
        curva, n_ops, u_final, atr_actual = ejecutar_simulacion_atr("QQQ")
        st.line_chart(curva)
        
        beneficio = curva[-1] - capital_total
        
        # AUTO-CORRECCIÓN
        if beneficio < 0 and n_ops > 2:
            st.session_state['auto_bias'] += 0.5
            st.error("Pérdida detectada. La IA sube el umbral ligeramente.")
        elif beneficio > 0:
            st.session_state['auto_bias'] = max(-5.0, st.session_state['auto_bias'] - 0.5)
            st.success("Beneficio detectado. El Escudo ATR absorbió el ruido del mercado correctamente.")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Operaciones", n_ops)
        c_b.metric("Beneficio Neto", f"{beneficio:.2f} €", delta="Línea base: -6.41 €")
        c_c.metric("Capital Final", f"{curva[-1]:.2f} €")
        
        st.info(f"💡 **Dato Técnico:** El QQQ tiene hoy una turbulencia base del **{atr_actual:.2f}%**. Con tu multiplicador de {multiplicador_atr}x, el paracaídas saltará si cae un **{(atr_actual * multiplicador_atr):.2f}%** desde el pico.")
