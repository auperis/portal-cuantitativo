# ==============================================================================
# ARQUITECTURA FASE 27: EL ESCUDO DINÁMICO (TRAILING STOP LOGIC)
# Objetivo: Permitir que las operaciones ganadoras crezcan con el tiempo 
# (Let Winners Run) y cerrar automáticamente si la tendencia se invierte.
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
st.set_page_config(page_title="Portal IA - Escudo Dinámico", layout="wide", page_icon="🛡️")

if 'auto_bias' not in st.session_state:
    st.session_state['auto_bias'] = 0.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN
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
st.sidebar.header("🛡️ Escudo Dinámico (Extracción)")
# NUEVO: Ajuste del Trailing Stop (La cuerda del paracaídas)
stop_loss_pct = st.sidebar.slider("Trailing Stop (%)", 1.0, 10.0, 4.0, help="Vende si el precio cae este % desde el máximo alcanzado.")

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
# 4. SIMULADOR CON TRAILING STOP (MÁQUINA DE ESTADOS)
# ------------------------------------------------------------------------------
def ejecutar_simulacion_realista(ticker, dias=30):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="2y"))
    
    # Variables de la "Máquina de Estados"
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo_alcanzado = 0
    
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    
    for i in range(len(df) - dias, len(df)):
        hoy = df.iloc[i]
        
        # 1. SI ESTAMOS DENTRO DEL GLOBO (Gestionar Extracción)
        if en_posicion:
            # Actualizamos la altura máxima alcanzada por el globo
            precio_maximo_alcanzado = max(precio_maximo_alcanzado, hoy['Close'])
            # Calculamos dónde está la cuerda de seguridad
            precio_stop_loss = precio_maximo_alcanzado * (1 - (stop_loss_pct / 100))
            
            # Si el globo cae por debajo de la cuerda, o es el último día de simulación: SALTAMOS
            if hoy['Close'] <= precio_stop_loss or i == len(df) - 1:
                valor_venta = acciones_compradas * hoy['Close']
                liquidez += (valor_venta - comision_fija) # Sumamos liquidez, pagamos comisión de venta
                en_posicion = False
                acciones_compradas = 0
                precio_maximo_alcanzado = 0
                
        # 2. SI ESTAMOS FUERA DEL GLOBO (Buscar Entrada)
        # Solo miramos si no es el último día (no tiene sentido comprar el último día)
        if not en_posicion and i < len(df) - 1:
            estudio = df.iloc[:i]
            prob = entrenar_ia(estudio)
            
            if prob >= umbral_final and hoy['Volatilidad'] > 0.5:
                ops += 1 # Registramos la operación
                en_posicion = True
                precio_maximo_alcanzado = hoy['Close']
                
                # Escalado Dinámico de la Apuesta
                rango_prob = 100 - umbral_final
                factor = min(max((prob - umbral_final) / rango_prob, 0), 1) if rango_prob > 0 else 0
                exposicion_pct = 0.05 + (factor * (max_exposicion/100 - 0.05))
                
                dinero_a_invertir = liquidez * exposicion_pct
                liquidez -= comision_fija # Pagamos comisión de entrada
                
                acciones_compradas = dinero_a_invertir / hoy['Close']
                liquidez -= dinero_a_invertir # Retiramos el dinero de la liquidez para comprar
                
        # 3. FOTOGRAFÍA DIARIA DEL PATRIMONIO
        # El patrimonio total es el dinero líquido MÁS lo que valen nuestras acciones hoy
        valor_cartera_hoy = liquidez + (acciones_compradas * hoy['Close']) if en_posicion else liquidez
        curva_capital.append(valor_cartera_hoy)
        
    return curva_capital, ops, umbral_final

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🛡️ Portal IA: El Escudo Dinámico (Trailing Stop)")

c1, c2, c3 = st.columns(3)
umbral_f = umbral_base + st.session_state['auto_bias']

with c1:
    st.metric("Umbral de Seguridad", f"{umbral_f:.1f}%", f"{st.session_state['auto_bias']}% IA Bias")
with c2:
    st.metric("Trailing Stop", f"-{stop_loss_pct}%", "Cuerda de seguridad")
with c3:
    if st.button("♻️ Reiniciar Aprendizaje"):
        st.session_state['auto_bias'] = 0.0
        st.rerun()

st.divider()

if st.button("🏁 Iniciar Simulación Avanzada (Multi-día)"):
    with st.spinner("Simulando entradas francotirador y salidas con Escudo Dinámico..."):
        curva, n_ops, u_final = ejecutar_simulacion_realista("QQQ")
        st.line_chart(curva)
        
        beneficio = curva[-1] - capital_total
        
        if beneficio < 0 and n_ops > 2:
            st.session_state['auto_bias'] += 1.0
            st.error("Pérdida detectada. La IA sube el umbral para ser más estricta.")
        elif beneficio > 5:
            st.session_state['auto_bias'] = max(-5.0, st.session_state['auto_bias'] - 0.5)
            st.success("Beneficio detectado. El Trailing Stop funcionó y capturó la tendencia.")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Operaciones (Trades cerrados)", n_ops)
        c_b.metric("Beneficio Neto (Comisiones descontadas)", f"{beneficio:.2f} €")
        c_c.metric("Capital Final", f"{curva[-1]:.2f} €")
