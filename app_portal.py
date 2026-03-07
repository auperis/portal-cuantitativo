# ==============================================================================
# ARQUITECTURA FASE 10: PERSISTENCIA Y EJECUCIÓN FRACCIONADA
# Objetivo: Resolver el error de '0 acciones' permitiendo decimales (Fractional Shares).
# Este código es el motor completo y visual para Streamlit.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="🎯")

st.title("🎯 Radar IA: Ejecución Fraccionada")
st.markdown("Optimización de capital de 1.000 € para activos de alto precio (ej. QQQ, SPY).")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL (Barra Lateral)
# ------------------------------------------------------------------------------
st.sidebar.header("Configuración del Radar")
# Lista de activos sugeridos para una cuenta de 1.000 €
tickers_input = st.sidebar.text_input("Lista de Activos (separados por coma)", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Calcular Órdenes")

# ------------------------------------------------------------------------------
# 3. MOTOR LÓGICO (IA + MATEMÁTICAS DE RIESGO)
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    """Descarga datos históricos de Yahoo Finance."""
    return yf.Ticker(ticker).history(period="3y")

def calcular_indicadores(df):
    """Calcula las pistas (features) para que la IA tome decisiones."""
    datos = df.copy()
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    datos['Volatilidad_5D'] = datos['Close'].rolling(window=5).std()
    datos['Media_Volumen_20D'] = datos['Volume'].rolling(window=20).mean()
    datos['Volumen_Relativo'] = datos['Volume'] / datos['Media_Volumen_20D']
    datos['Retorno_3D_%'] = datos['Close'].pct_change(periods=3) * 100
    
    # Cálculo del RSI (La Goma Elástica)
    delta = datos['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    ema_down = ema_down.replace(0, 0.001) # Evitar división por cero
    datos['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))
    
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    return datos.dropna()

def entrenar_modelo(df):
    """Entrena el cerebro de la IA para cada activo del radar."""
    pistas = ['Retorno_Hoy_%', 'Volatilidad_5D', 'Volumen_Relativo', 'Retorno_3D_%', 'RSI_14']
    corte = int(len(df) * 0.8)
    estudio, examen = df.iloc[:corte], df.iloc[corte:]
    
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(estudio[pistas], estudio['Target_Mañana_Sube'])
    
    prob = modelo.predict_proba(examen[pistas])[:, 1] * 100
    precision = accuracy_score(examen['Target_Mañana_Sube'], np.where(prob >= 50, 1, 0)) * 100
    return modelo, precision, pistas

# FUNCIÓN CRÍTICA: Ahora permite comprar trozos de acciones (decimales)
def calcular_ejecucion_fraccionada(capital, precio, stop_loss, riesgo_max=2.0):
    """Calcula cuántas acciones comprar basándose en el riesgo máximo permitido (20€)."""
    perdida_max_euros = capital * (riesgo_max / 100)
    riesgo_por_accion = precio * (stop_loss / 100)
    
    if riesgo_por_accion <= 0: return 0, 0, 0
    
    # Permitimos decimales para activos caros
    acciones_a_comprar = perdida_max_euros / riesgo_por_accion
    
    # Límite de seguridad: No invertir más del capital total disponible
    if (acciones_a_comprar * precio) > capital:
        acciones_a_comprar = capital / precio
        
    capital_total_invertido = acciones_a_comprar * precio
    return acciones_a_comprar, capital_total_invertido, perdida_max_euros

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN DEL PORTAL
# ------------------------------------------------------------------------------
if boton_analizar:
    lista_activos = [x.strip().upper() for x in tickers_input.split(',')]
    resultados = []
    
    progreso = st.progress(0)
    for i, ticker in enumerate(lista_activos):
        try:
            df_raw = obtener_datos(ticker)
            if not df_raw.empty:
                df = calcular_indicadores(df_raw)
                mod, prec, pists = entrenar_modelo(df)
                
                # Predicción actual
                hoy = df.iloc[-1:]
                prob = mod.predict_proba(hoy[pists])[0][1] * 100
                precio = hoy['Close'].values[0]
                
                resultados.append({
                    "Activo": ticker,
                    "Convicción (%)": round(prob, 2),
                    "Precisión (%)": round(prec, 2),
                    "Precio ($)": round(precio, 2)
                })
        except Exception as e: 
            st.warning(f"Error procesando {ticker}: {e}")
        progreso.progress((i + 1) / len(lista_activos))

    if resultados:
        df_rank = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
        st.subheader("🏆 Ranking de Oportunidades")
        st.dataframe(df_rank.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), use_container_width=True)
        
        ganador = df_rank.iloc[0]
        if ganador["Convicción (%)"] >= umbral_conviccion:
            st.success(f"👑 ACTIVO ELEGIDO: {ganador['Activo']} con {ganador['Convicción (%)']}% de convicción.")
            
            # Cálculo de la orden con decimales
            num_acciones, inv_total, riesgo = calcular_ejecucion_fraccionada(
                capital_usuario, ganador["Precio ($)"], stop_loss_usuario
            )
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Acciones (Decimales)", f"{num_acciones:.4f}")
            col2.metric("Inversión en Euros", f"{inv_total:.2f} €")
            col3.metric("Riesgo Máximo", f"{riesgo:.2f} €")
            
            st.info(f"👉 **Instrucción al Broker:** Compra **{num_acciones:.4f}** acciones de {ganador['Activo']}. Broker sugerido: Revolut o Interactive Brokers.")
        else:
            st.error(f"Ningún activo supera el filtro del {umbral_conviccion}%. MANTENER LIQUIDEZ.")
