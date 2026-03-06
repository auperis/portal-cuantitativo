# ==============================================================================
# ARQUITECTURA FASE 4: EL PORTAL COMPLETO CON GESTIÓN DE RIESGO
# Objetivo: IA Predictiva + Escudo Matemático de Capital
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
ticker_usuario = st.sidebar.text_input("Símbolo del Activo (ej. SPY, AAPL, BTC-USD)", value="SPY")

# CONTROLES DE RIESGO AÑADIDOS
st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, max_value=10000, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

boton_analizar = st.sidebar.button("Ejecutar Oráculo IA")

# ------------------------------------------------------------------------------
# 3. EL MOTOR OCULTO Y EL ESCUDO MATEMÁTICO
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

# Aquí está tu Módulo 6 integrado en las tripas del portal
def calcular_tamaño_posicion(capital_total, precio_accion, stop_loss_porcentaje, riesgo_maximo_porcentaje=2.0):
    riesgo_en_euros = capital_total * (riesgo_maximo_porcentaje / 100)
    riesgo_por_accion = precio_accion * (stop_loss_porcentaje / 100)
    
    if riesgo_por_accion <= 0:
        return 0, 0, 0
        
    numero_acciones = int(riesgo_en_euros / riesgo_por_accion)
    capital_a_invertir = numero_acciones * precio_accion
    return numero_acciones, capital_a_invertir, riesgo_en_euros

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB (Lo que ocurre al pulsar el botón)
# ------------------------------------------------------------------------------
if boton_analizar:
    with st.spinner(f"Extrayendo datos de {ticker_usuario} y entrenando red neuronal..."):
        
        datos_crudos = obtener_datos(ticker_usuario)
        
        if datos_crudos.empty:
            st.error("Error: Activo no encontrado. Comprueba el símbolo.")
        else:
            datos_procesados = calcular_indicadores(datos_crudos)
            modelo, precision_ia, pistas = entrenar_modelo(datos_procesados)
            
            # --- ZONA VISUAL DE LA WEB ---
            st.metric(label="Precisión Histórica del Modelo (Edge)", value=f"{precision_ia:.2f}%")
            
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
            
            precio_actual = datos_hoy['Close'].values[0]
            
            if prediccion_mañana == 1:
                st.success("🟢 LUZ VERDE: Probabilidad matemática de subida. Entorno favorable para desplegar capital.")
                
                # LA TRADUCCIÓN VISUAL DE TU MÓDULO 6 A LA WEB
                st.subheader("🛡️ Instrucciones de Ejecución (Broker)")
                acciones, inversion, riesgo_max = calcular_tamaño_posicion(
                    capital_usuario, precio_actual, stop_loss_usuario
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Acciones a Comprar", f"{acciones}")
                col2.metric("Capital a Invertir", f"{inversion:.2f} €")
                col3.metric("Riesgo Máximo", f"{riesgo_max:.2f} €")
                
                st.info(f"👉 **Orden sugerida:** Compra {acciones} acciones de {ticker_usuario}. Configura tu Stop-Loss automático a un -{stop_loss_usuario}% de caída. Si la IA falla, el escudo te expulsará del mercado con una pérdida máxima de {riesgo_max:.2f} €.")
                
            else:
                st.error("🔴 LUZ ROJA: Probabilidad matemática de caída. Mantener liquidez, proteger capital.")
