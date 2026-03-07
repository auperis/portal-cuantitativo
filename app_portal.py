# ==============================================================================
# ARQUITECTURA FASE 31: CALIBRACIÓN DE PROBABILIDADES (PROBABILITY CALIBRATION)
# Objetivo: Medir la confianza real de la IA en predicciones a largo plazo
# para ajustar el umbral de entrada de forma realista.
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
st.set_page_config(page_title="Portal IA - Calibración", layout="wide", page_icon="🎛️")

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
st.sidebar.header("🎛️ Calibración de la IA")
dias_vision_ia = st.sidebar.slider("Horizonte de Predicción IA (Días)", 1, 15, 5)

# ATENCIÓN: Hemos bajado el rango del slider para permitir umbrales más realistas (desde 50%)
# Para predecir a 5 días, un 53% puede ser una ventaja estadística masiva.
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 70.0, 52.0, 
                                help="Para Swing Trading, un 52-54% es a menudo el 'Punto Dulce'.")

st.sidebar.divider()
st.sidebar.header("⏱️ El Escudo y Tiempo")
dias_simulacion = st.sidebar.selectbox("Días de Simulación", [30, 90, 180, 365], index=2)
multiplicador_atr = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0)
filtro_macro = st.sidebar.checkbox("Activar Lente Macro (Media 200)", value=True)

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

def entrenar_ia(df_hist, dias_vision):
    df = df_hist.copy()
    df['Target'] = np.where(df['Close'].shift(-dias_vision) > df['Close'] * 1.01, 1, 0)
    df = df.dropna()
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    
    if len(df) < 50:
        return 50.0 
        
    # Aumentamos ligeramente la profundidad del bosque para encontrar patrones más complejos a largo plazo
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(df[pistas], df['Target'])
    
    prob = model.predict_proba(df_hist[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR FASE 31 (CON ESCÁNER DE CONVICCIÓN)
# ------------------------------------------------------------------------------
def ejecutar_simulacion_calibracion(ticker, dias, dias_vision):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="3y"))
    
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo_alcanzado = 0
    
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    # NUEVO: Rastreador de Convicción
    pico_maximo_probabilidad = 0.0
    
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
        if not en_posicion and i < len(df) - dias_vision:
            estudio = df.iloc[:i]
            prob = entrenar_ia(estudio, dias_vision)
            
            # Registramos el pico máximo alcanzado por la IA
            if prob > pico_maximo_probabilidad:
                pico_maximo_probabilidad = prob
            
            pasa_macro = (hoy['Close'] > hoy['Media_200']) if filtro_macro else True
            
            if prob >= umbral_final and hoy['Volatilidad'] > 0.5 and pasa_macro and hoy['RSI'] < 70:
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
        
    return curva_capital, ops, umbral_final, pico_maximo_probabilidad

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🎛️ Portal IA: Calibración de Convicción")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Decisión Actual", f"{umbral_f:.1f}%")
with c2:
    st.metric("Visión de la IA", f"{dias_vision_ia} Días al futuro")
with c3:
    if st.button("♻️ Reiniciar Memoria IA"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button(f"🏁 Iniciar Calibración (Simulación {dias_simulacion} Días)"):
    with st.spinner("Escaneando las probabilidades ocultas de la IA..."):
        curva, n_ops, u_final, pico_prob = ejecutar_simulacion_calibracion("QQQ", dias_simulacion, dias_vision_ia)
        
        st.line_chart(curva)
        beneficio = curva[-1] - capital_total
        
        c_a, c_b, c_c, c_d = st.columns(4)
        c_a.metric("Operaciones", n_ops)
        c_b.metric("Beneficio Neto", f"{beneficio:.2f} €")
        c_c.metric("Capital Final", f"{curva[-1]:.2f} €")
        
        # EL DATO VITAL: El Pico Máximo de Convicción
        c_d.metric("Pico Máx. de IA", f"{pico_prob:.1f}%", delta="Confianza Real", delta_color="normal")
        
        st.divider()
        if n_ops == 0:
            st.error(f"⚠️ PARÁLISIS: La IA nunca superó el {umbral_f}%. Su pico máximo de confianza en 180 días fue solo del {pico_prob:.1f}%.")
            st.info(f"🛠️ **Solución del Arquitecto:** Baja el 'Umbral Probabilidad (%)' en la barra lateral a un valor ligeramente por debajo de {pico_prob:.1f}% (por ejemplo, {pico_prob - 1.0:.1f}%) y vuelve a simular.")
        elif beneficio > 0:
            st.success("✅ ¡Calibración Perfecta! Hemos encontrado el punto dulce para hacer Swing Trading institucional.")
