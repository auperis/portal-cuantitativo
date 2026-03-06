# ==============================================================================
# ARQUITECTURA FASE 2: EL ESCAPARATE WEB (STREAMLIT)
# Objetivo: Convertir nuestro motor de IA en una aplicación web interactiva.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA (La "Pintura" de la Carrocería)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📈")

st.title("🤖 Portal de Inteligencia Cuantitativa")
st.markdown("Plataforma de análisis predictivo para carteras eficientes.")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL (El "Volante" para el usuario)
# ------------------------------------------------------------------------------
st.sidebar.header("Parámetros de Inversión")
# El usuario escribe el nombre de la acción en una caja de texto
ticker_usuario = st.sidebar.text_input("Símbolo del Activo (ej. SPY, AAPL, BTC-USD)", value="SPY")
# El usuario pulsa un botón para iniciar la IA
boton_analizar = st.sidebar.button("Ejecutar Oráculo IA")

# ------------------------------------------------------------------------------
# 3. EL MOTOR OCULTO (Las funciones que ya construimos en Colab)
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    activo = yf.Ticker(ticker)
    return activo.history(period="2y")

def calcular_indicadores(df):
    datos = df.copy()
    datos['Media_20_Dias'] = datos['Close'].rolling(window=20).mean()
    datos['Distancia_a_Media_%'] = ((datos['Close'] / datos['Media_20_Dias']) - 1) * 100
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    return datos.dropna()

def entrenar_modelo(df):
    columnas_pistas = ['Distancia_a_Media_%', 'Retorno_Hoy_%', 'Volume']
    indice_corte = int(len(df) * 0.8)
    datos_estudio = df.iloc[:indice_corte]
    datos_examen = df.iloc[indice_corte:]
    
    modelo_ia = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_ia.fit(datos_estudio[columnas_pistas], datos_estudio['Target_Mañana_Sube'])
    
    predicciones = modelo_ia.predict(datos_examen[columnas_pistas])
    precision = accuracy_score(datos_examen['Target_Mañana_Sube'], predicciones) * 100
    return modelo_ia, precision, columnas_pistas

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB (Lo que ocurre al pulsar el botón)
# ------------------------------------------------------------------------------
if boton_analizar:
    with st.spinner(f"Extrayendo datos de {ticker_usuario} y entrenando red neuronal..."):
        # Ejecutamos el motor paso a paso
        datos_crudos = obtener_datos(ticker_usuario)
        
        if datos_crudos.empty:
            st.error("Error: Activo no encontrado. Comprueba el símbolo.")
        else:
            datos_procesados = calcular_indicadores(datos_crudos)
            modelo, precision_ia, pistas = entrenar_modelo(datos_procesados)
            
            # --- ZONA VISUAL DE LA WEB ---
            
            # Mostramos la precisión como una métrica de negocio
            st.metric(label="Precisión Histórica del Modelo (Edge)", value=f"{precision_ia:.2f}%")
            
            # Generamos la Radiografía (Gráfico)
            st.subheader("Radiografía del Mercado")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=datos_procesados.index, y=datos_procesados['Close'], name='Precio'))
            fig.add_trace(go.Scatter(x=datos_procesados.index, y=datos_procesados['Media_20_Dias'], name='Radar 20D', line=dict(dash='dot')))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # El Oráculo (Semáforo)
            st.subheader("Señal para Mañana")
            datos_hoy = datos_procesados.iloc[-1:]
            prediccion_mañana = modelo.predict(datos_hoy[pistas])[0]
            
            if prediccion_mañana == 1:
                st.success("🟢 LUZ VERDE: Probabilidad matemática de subida. Entorno favorable para desplegar capital.")
            else:
                st.error("🔴 LUZ ROJA: Probabilidad matemática de caída. Mantener liquidez, proteger capital.")
