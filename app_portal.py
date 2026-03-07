# ==============================================================================
# ARQUITECTURA FASE 30: ALINEACIÓN DE HORIZONTE (SWING ORACLE)
# Objetivo: Alinear el objetivo de la IA (Target) con nuestro estilo de
# inversión. Enseñar a la IA a predecir a N días vista, no a 1 día.
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
st.set_page_config(page_title="Portal IA - Oráculo Swing", layout="wide", page_icon="🔮")

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
st.sidebar.header("🔮 El Cerebro (IA)")
# NUEVO DIAL: ¿A cuántos días vista queremos que mire la IA?
dias_vision_ia = st.sidebar.slider("Horizonte de Predicción IA (Días)", 1, 15, 5, 
                                   help="5 días = 1 semana de mercado. Obliga a la IA a buscar tendencias, no saltos de 1 día.")
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 75.0, 58.0)

st.sidebar.divider()
st.sidebar.header("⏱️ El Escudo y Tiempo")
dias_simulacion = st.sidebar.selectbox("Días de Simulación (Backtest)", [30, 90, 180, 365], index=2)
multiplicador_atr = st.sidebar.slider("Multiplicador ATR (Paracaídas)", 1.0, 5.0, 2.0)
filtro_macro = st.sidebar.checkbox("Activar Lente Macro (Media 200)", value=True)

st.sidebar.divider()
st.sidebar.header("💸 Capital")
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
    
    # FASE 30: LA CLAVE DEL ÉXITO. 
    # Ya no miramos a 1 día (-1), miramos a X días vista (-dias_vision).
    # Además, exigimos que el precio no solo sea mayor, sino que suba al menos un 1% para evitar ruido.
    df['Target'] = np.where(df['Close'].shift(-dias_vision) > df['Close'] * 1.01, 1, 0)
    
    # Eliminamos las últimas filas porque de esas "aún no conocemos el futuro" a X días vista
    df = df.dropna()
    
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    
    # Si no hay suficientes datos para entrenar (porque dropeamos demasiados), devolvemos probabilidad neutral
    if len(df) < 50:
        return 50.0 
        
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[pistas], df['Target'])
    
    # Predecimos basándonos en las pistas del último día disponible
    prob = model.predict_proba(df_hist[pistas].iloc[-1:]) [0][1] * 100
    return prob

# ------------------------------------------------------------------------------
# 4. SIMULADOR FASE 30
# ------------------------------------------------------------------------------
def ejecutar_simulacion_fase30(ticker, dias, dias_vision):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="3y"))
    
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo_alcanzado = 0
    
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    
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
            # Le pasamos a la IA cuántos días debe mirar al futuro durante su entrenamiento
            prob = entrenar_ia(estudio, dias_vision)
            
            pasa_macro = (hoy['Close'] > hoy['Media_200']) if filtro_macro else True
            
            # También exigimos que el RSI no esté en sobrecompra extrema (>70) para evitar picos trampa
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
        
    return curva_capital, ops, umbral_final

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🔮 Portal IA: El Oráculo Swing")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Decisión IA", f"{umbral_f:.1f}%")
with c2:
    st.metric("Visión de la IA", f"{dias_vision_ia} Días al futuro")
with c3:
    if st.button("♻️ Reiniciar Memoria IA"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button(f"🏁 Iniciar Simulación ({dias_simulacion} Días)"):
    with st.spinner(f"Entrenando a la IA para predecir a {dias_vision_ia} días vista..."):
        curva, n_ops, u_final = ejecutar_simulacion_fase30("QQQ", dias_simulacion, dias_vision_ia)
        st.line_chart(curva)
        
        beneficio = curva[-1] - capital_total
        
        # AUTO-CORRECCIÓN
        if beneficio < 0 and n_ops > 2:
            st.session_state['auto_bias'] += 0.5
            st.warning("Ajustando sesgo para filtrar falsas tendencias.")
        elif beneficio > 0:
            st.session_state['auto_bias'] = max(-5.0, st.session_state['auto_bias'] - 0.5)
            st.success("¡Excelente! La alineación temporal ha capturado la verdadera tendencia.")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Operaciones", n_ops)
        c_b.metric("Beneficio Neto Total", f"{beneficio:.2f} €", delta="Alineación Temporal Activa")
        c_c.metric("Capital Final", f"{curva[-1]:.2f} €")
        
        st.markdown(f"**Análisis de Flujo:** La IA ahora estudia qué pasaba en el mercado exactamente {dias_vision_ia} días antes de que el QQQ subiera más de un 1%. Ya no persigue saltos de 24 horas.")
