# ==============================================================================
# ARQUITECTURA FASE 6.1: EXPLAINABLE AI (XAI)
# Objetivo: Auditar el cerebro de la IA para detectar qué datos son ruido.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📈")

st.title("🤖 Portal de Inteligencia Cuantitativa")
st.markdown("Plataforma de análisis predictivo y backtesting para carteras eficientes.")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL (Barra Lateral)
# ------------------------------------------------------------------------------
st.sidebar.header("Parámetros de Inversión")
ticker_usuario = st.sidebar.text_input("Símbolo del Activo (ej. SPY, AAPL, BTC-USD)", value="SPY")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, max_value=10000, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

boton_analizar = st.sidebar.button("Ejecutar Oráculo y Simulador IA")

# ------------------------------------------------------------------------------
# 3. EL MOTOR OCULTO Y EL ESCUDO
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    activo = yf.Ticker(ticker)
    return activo.history(period="3y")

def calcular_indicadores(df):
    datos = df.copy()
    
    # Pistas Base
    datos['Media_20_Dias'] = datos['Close'].rolling(window=20).mean()
    datos['Distancia_a_Media_%'] = ((datos['Close'] / datos['Media_20_Dias']) - 1) * 100
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    
    # Nuevas Pistas Institucionales
    datos['Volatilidad_5D'] = datos['Close'].rolling(window=5).std()
    datos['Media_Volumen_20D'] = datos['Volume'].rolling(window=20).mean()
    datos['Volumen_Relativo'] = datos['Volume'] / datos['Media_Volumen_20D']
    datos['Retorno_3D_%'] = datos['Close'].pct_change(periods=3) * 100
    
    # Target
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    
    return datos.dropna()

def entrenar_modelo(df):
    columnas_pistas = [
        'Distancia_a_Media_%', 
        'Retorno_Hoy_%', 
        'Volatilidad_5D', 
        'Volumen_Relativo', 
        'Retorno_3D_%'
    ]
    
    indice_corte = int(len(df) * 0.8)
    datos_estudio = df.iloc[:indice_corte]
    datos_examen = df.iloc[indice_corte:]
    
    modelo_ia = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_ia.fit(datos_estudio[columnas_pistas], datos_estudio['Target_Mañana_Sube'])
    
    predicciones_examen = modelo_ia.predict(datos_examen[columnas_pistas])
    precision = accuracy_score(datos_examen['Target_Mañana_Sube'], predicciones_examen) * 100
    
    # NUEVO: Extraemos la importancia matemática de cada pista (El polígrafo)
    importancias = modelo_ia.feature_importances_ * 100
    
    return modelo_ia, precision, columnas_pistas, datos_examen, predicciones_examen, importancias

def calcular_tamaño_posicion(capital_total, precio_accion, stop_loss_porcentaje, riesgo_maximo_porcentaje=2.0):
    riesgo_en_euros = capital_total * (riesgo_maximo_porcentaje / 100)
    riesgo_por_accion = precio_accion * (stop_loss_porcentaje / 100)
    
    if riesgo_por_accion <= 0: return 0, 0, 0
        
    numero_acciones = int(riesgo_en_euros / riesgo_por_accion)
    capital_a_invertir = numero_acciones * precio_accion
    return numero_acciones, capital_a_invertir, riesgo_en_euros

def ejecutar_simulador(datos_examen, predicciones_examen, capital_inicial):
    df_sim = datos_examen.copy()
    df_sim['Señal_IA'] = predicciones_examen
    
    df_sim['Señal_Ayer'] = df_sim['Señal_IA'].shift(1)
    df_sim['Retorno_IA_%'] = np.where(df_sim['Señal_Ayer'] == 1, df_sim['Retorno_Hoy_%'], 0)
    
    df_sim['Capital_Inversor_Tradicional'] = capital_inicial * (1 + (df_sim['Retorno_Hoy_%'] / 100)).cumprod()
    df_sim['Capital_Estrategia_IA'] = capital_inicial * (1 + (df_sim['Retorno_IA_%'] / 100)).cumprod()
    
    return df_sim.dropna()

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB 
# ------------------------------------------------------------------------------
if boton_analizar:
    with st.spinner(f"Viajando en el tiempo para simular cartera con {ticker_usuario}..."):
        
        datos_crudos = obtener_datos(ticker_usuario)
        
        if datos_crudos.empty:
            st.error("Error: Activo no encontrado.")
        else:
            datos_procesados = calcular_indicadores(datos_crudos)
            # Recibimos las importancias del Cerebro
            modelo, precision_ia, pistas, datos_examen, predic_examen, importancias = entrenar_modelo(datos_procesados)
            
            st.info(f"🧠 Cerebro IA analizando {len(pistas)} variables predictivas simultáneamente.")
            st.metric(label="Precisión Histórica del Modelo (Edge)", value=f"{precision_ia:.2f}%")
            
            # --- NUEVA ZONA: AUDITORÍA XAI ---
            st.subheader("🔍 Auditoría de IA (Explainable AI - XAI)")
            st.markdown("El Polígrafo: ¿Qué pistas son útiles y cuáles son puro ruido?")
            
            df_importancias = pd.DataFrame({'Pista': pistas, 'Importancia %': importancias})
            df_importancias = df_importancias.sort_values(by='Importancia %', ascending=True)
            
            fig_xai = go.Figure(go.Bar(
                x=df_importancias['Importancia %'],
                y=df_importancias['Pista'],
                orientation='h',
                marker_color='#00d2d3' # Color cian tecnológico
            ))
            fig_xai.update_layout(template='plotly_dark', height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Importancia en la Decisión (%)")
            st.plotly_chart(fig_xai, use_container_width=True)
            
            st.markdown("---")
            
            # --- ZONA DEL SIMULADOR ---
            st.subheader("✈️ Simulador de Vuelo: IA vs Inversor Tradicional")
            
            df_simulado = ejecutar_simulador(datos_examen, predic_examen, capital_usuario)
            
            capital_final_tradicional = df_simulado['Capital_Inversor_Tradicional'].iloc[-1]
            capital_final_ia = df_simulado['Capital_Estrategia_IA'].iloc[-1]
            
            col_sim1, col_sim2 = st.columns(2)
            col_sim1.metric("Cuenta: Inversor Tradicional", f"{capital_final_tradicional:.2f} €")
            col_sim2.metric("Cuenta: Inteligencia Cuantitativa (IA)", f"{capital_final_ia:.2f} €", 
                            delta=f"Diferencia: {(capital_final_ia - capital_final_tradicional):.2f} €")
            
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=df_simulado.index, y=df_simulado['Capital_Inversor_Tradicional'], 
                                         name='Tradicional', line=dict(color='gray')))
            fig_sim.add_trace(go.Scatter(x=df_simulado.index, y=df_simulado['Capital_Estrategia_IA'], 
                                         name='Estrategia IA', line=dict(color='green', width=3)))
            fig_sim.update_layout(template='plotly_dark', height=350, yaxis_title='Capital en Euros (€)')
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.markdown("---")
            
            # --- ZONA DEL ORÁCULO ---
            st.subheader("Señal para Mañana (Tiempo Real)")
            datos_hoy = datos_procesados.iloc[-1:]
            prediccion_mañana = modelo.predict(datos_hoy[pistas])[0]
            precio_actual = datos_hoy['Close'].values[0]
            
            if prediccion_mañana == 1:
                st.success("🟢 LUZ VERDE: Probabilidad matemática de subida. Entorno favorable.")
                st.subheader("🛡️ Instrucciones de Ejecución (Broker)")
                acciones, inversion, riesgo_max = calcular_tamaño_posicion(capital_usuario, precio_actual, stop_loss_usuario)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Acciones a Comprar", f"{acciones}")
                col2.metric("Capital a Invertir", f"{inversion:.2f} €")
                col3.metric("Riesgo Máximo", f"{riesgo_max:.2f} €")
                st.info(f"👉 **Orden sugerida:** Compra {acciones} acciones de {ticker_usuario}. Stop-Loss al -{stop_loss_usuario}%. Pérdida máxima bloqueada en {riesgo_max:.2f} €.")
            else:
                st.error("🔴 LUZ ROJA: Probabilidad matemática de caída. Mantener liquidez, proteger capital.")
