# ==============================================================================
# ARQUITECTURA FASE 29: RÉGIMEN MACRO Y EXPANSIÓN TEMPORAL
# Objetivo: Ampliar el horizonte de simulación para dejar correr ganancias y 
# añadir un filtro de Régimen Macro (SMA 200) para evitar inviernos financieros.
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
st.set_page_config(page_title="Portal IA - Régimen Macro", layout="wide", page_icon="🔭")

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
st.sidebar.header("⏱️ Horizonte Temporal")
# NUEVO: Expansión del tiempo de simulación
dias_simulacion = st.sidebar.selectbox("Días de Simulación (Backtest)", [30, 90, 180, 365], index=1,
                                       help="90 o 180 días permite ver cómo maduran las operaciones institucionales.")

st.sidebar.divider()
st.sidebar.header("🔭 Filtros de Precisión")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0)
multiplicador_atr = st.sidebar.slider("Multiplicador ATR (Paracaídas)", 1.0, 5.0, 2.0)

# NUEVO: Filtro de Régimen Macro
filtro_macro = st.sidebar.checkbox("Activar Lente Macro (Media 200)", value=True,
                                   help="Prohíbe comprar si el activo está en tendencia bajista de largo plazo.")

st.sidebar.divider()
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)
max_exposicion = st.sidebar.slider("Exposición Máxima (%)", 5.0, 40.0, 25.0)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_50'] = d['Close'].rolling(50).mean()
    
    # NUEVO: La línea que separa el Verano del Invierno Financiero
    d['Media_200'] = d['Close'].rolling(200).mean() 
    
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    d['High_Low'] = d['High'] - d['Low']
    d['High_PrevClose'] = np.abs(d['High'] - d['Close'].shift(1))
    d['Low_PrevClose'] = np.abs(d['Low'] - d['Close'].shift(1))
    d['True_Range'] = d[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    d['ATR'] = d['True_Range'].rolling(14).mean()
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
# 4. SIMULADOR FASE 29 (MACRO + ATR)
# ------------------------------------------------------------------------------
def ejecutar_simulacion_fase29(ticker, dias):
    # Descargamos más tiempo (3y) para asegurarnos de que la Media 200 tiene datos
    df = calcular_indicadores(yf.Ticker(ticker).history(period="3y"))
    
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo_alcanzado = 0
    
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    # Recorremos el número de días seleccionado en el UI
    for i in range(len(df) - dias, len(df)):
        hoy = df.iloc[i]
        
        # 1. GESTIONAR EXTRACCIÓN
        if en_posicion:
            precio_maximo_alcanzado = max(precio_maximo_alcanzado, hoy['Close'])
            margen_permitido_pct = hoy['ATR_pct'] * multiplicador_atr
            precio_stop_loss = precio_maximo_alcanzado * (1 - (margen_permitido_pct / 100))
            
            if hoy['Close'] <= precio_stop_loss or i == len(df) - 1:
                valor_venta = acciones_compradas * hoy['Close']
                liquidez += (valor_venta - comision_fija)
                en_posicion = False
                acciones_compradas = 0
                precio_maximo_alcanzado = 0
                
        # 2. BUSCAR ENTRADA
        if not en_posicion and i < len(df) - 1:
            estudio = df.iloc[:i]
            prob = entrenar_ia(estudio)
            
            # EL LENTE MACRO: Solo operamos si estamos en Verano (Precio > Media 200)
            pasa_macro = (hoy['Close'] > hoy['Media_200']) if filtro_macro else True
            
            if prob >= umbral_final and hoy['Volatilidad'] > 0.5 and pasa_macro:
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
        
    return curva_capital, ops, umbral_final

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🔭 Portal IA: Lente Macro y Expansión")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Decisión IA", f"{umbral_f:.1f}%")
with c2:
    st.metric("Horizonte de Análisis", f"{dias_simulacion} Días")
with c3:
    if st.button("♻️ Reiniciar Memoria IA"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button(f"🏁 Iniciar Simulación ({dias_simulacion} Días)"):
    with st.spinner("Analizando regímenes de mercado y extendiendo el horizonte..."):
        curva, n_ops, u_final = ejecutar_simulacion_fase29("QQQ", dias_simulacion)
        st.line_chart(curva)
        
        beneficio = curva[-1] - capital_total
        
        # AUTO-CORRECCIÓN MACRO
        if beneficio < 0 and n_ops > 3:
            st.session_state['auto_bias'] += 0.5
            st.error("Rendimiento negativo detectado. Aumentando la severidad de los filtros.")
        elif beneficio > 0:
            st.session_state['auto_bias'] = max(-5.0, st.session_state['auto_bias'] - 0.5)
            st.success("¡Tendencia capturada con éxito! El sistema es rentable en este horizonte.")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Operaciones Completadas", n_ops)
        c_b.metric("Beneficio Neto Total", f"{beneficio:.2f} €", delta="Optimizado")
        c_c.metric("Capital Final", f"{curva[-1]:.2f} €")
