# ==============================================================================
# ARQUITECTURA FASE 8: OPTIMIZACIÓN DE FRECUENCIA (FILTRO DE CONVICCIÓN)
# Objetivo: Extraer la probabilidad interna de la IA para operar menos, pero mejor.
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

st.sidebar.header("🛡️ Gestión de Riesgo y Convicción")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, max_value=10000, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

# NUEVO: El control del peaje. Exigimos un nivel mínimo de seguridad a la IA.
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", min_value=50, max_value=80, value=55, step=1, 
                                      help="Si la IA no supera este % de seguridad, se quedará en Luz Roja (Liquidez) para ahorrar comisiones.")

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
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    datos['Volatilidad_5D'] = datos['Close'].rolling(window=5).std()
    datos['Media_Volumen_20D'] = datos['Volume'].rolling(window=20).mean()
    datos['Volumen_Relativo'] = datos['Volume'] / datos['Media_Volumen_20D']
    datos['Retorno_3D_%'] = datos['Close'].pct_change(periods=3) * 100
    
    # Oscilador RSI
    delta = datos['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    datos['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Target
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    
    return datos.dropna()

def entrenar_modelo(df):
    columnas_pistas = ['Retorno_Hoy_%', 'Volatilidad_5D', 'Volumen_Relativo', 'Retorno_3D_%', 'RSI_14']
    
    indice_corte = int(len(df) * 0.8)
    datos_estudio = df.iloc[:indice_corte]
    datos_examen = df.iloc[indice_corte:]
    
    modelo_ia = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_ia.fit(datos_estudio[columnas_pistas], datos_estudio['Target_Mañana_Sube'])
    
    # NUEVO: En lugar de predecir 1 o 0, le pedimos la probabilidad exacta (predict_proba)
    probabilidades_examen = modelo_ia.predict_proba(datos_examen[columnas_pistas])
    
    # Guardamos la probabilidad de que SUBE (columna 1)
    prob_subir_array = probabilidades_examen[:, 1] * 100
    
    # Calculamos la precisión base (usando el umbral normal del 50%) para mantener nuestra métrica histórica
    predicciones_base = np.where(prob_subir_array >= 50.0, 1, 0)
    precision = accuracy_score(datos_examen['Target_Mañana_Sube'], predicciones_base) * 100
    
    importancias = modelo_ia.feature_importances_ * 100
    
    return modelo_ia, precision, columnas_pistas, datos_examen, prob_subir_array, importancias

def calcular_tamaño_posicion(capital_total, precio_accion, stop_loss_porcentaje, riesgo_maximo_porcentaje=2.0):
    riesgo_en_euros = capital_total * (riesgo_maximo_porcentaje / 100)
    riesgo_por_accion = precio_accion * (stop_loss_porcentaje / 100)
    
    if riesgo_por_accion <= 0: return 0, 0, 0
        
    numero_acciones = int(riesgo_en_euros / riesgo_por_accion)
    capital_a_invertir = numero_acciones * precio_accion
    return numero_acciones, capital_a_invertir, riesgo_en_euros

def ejecutar_simulador(datos_examen, prob_subir_array, capital_inicial, umbral_conviccion):
    df_sim = datos_examen.copy()
    
    # NUEVO: El simulador solo invierte si la IA supera nuestro Filtro de Convicción
    df_sim['Señal_IA'] = np.where(prob_subir_array >= umbral_conviccion, 1, 0)
    
    df_sim['Señal_Ayer'] = df_sim['Señal_IA'].shift(1)
    df_sim['Retorno_IA_%'] = np.where(df_sim['Señal_Ayer'] == 1, df_sim['Retorno_Hoy_%'], 0)
    
    df_sim['Capital_Inversor_Tradicional'] = capital_inicial * (1 + (df_sim['Retorno_Hoy_%'] / 100)).cumprod()
    df_sim['Capital_Estrategia_IA'] = capital_inicial * (1 + (df_sim['Retorno_IA_%'] / 100)).cumprod()
    
    return df_sim.dropna()

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB 
# ------------------------------------------------------------------------------
if boton_analizar:
    with st.spinner(f"Entrenando modelo y aplicando filtro de convicción del {umbral_conviccion}%..."):
        
        datos_crudos = obtener_datos(ticker_usuario)
        
        if datos_crudos.empty:
            st.error("Error: Activo no encontrado.")
        else:
            datos_procesados = calcular_indicadores(datos_crudos)
            modelo, precision_ia, pistas, datos_examen, prob_subir_array, importancias = entrenar_modelo(datos_procesados)
            
            st.info(f"🧠 Cerebro IA analizando {len(pistas)} variables. Filtro de convicción activo: {umbral_conviccion}%.")
            
            if precision_ia > 52:
                st.metric(label="Precisión Histórica del Modelo Base", value=f"{precision_ia:.2f}%", delta="Ventaja Estadística")
            else:
                st.metric(label="Precisión Histórica del Modelo Base", value=f"{precision_ia:.2f}%", delta="Ruido de Mercado", delta_color="inverse")
            
            # --- ZONA DEL SIMULADOR ---
            st.subheader(f"✈️ Simulador de Vuelo (Operando solo con > {umbral_conviccion}% de seguridad)")
            
            # Pasamos el umbral al simulador
            df_simulado = ejecutar_simulador(datos_examen, prob_subir_array, capital_usuario, umbral_conviccion)
            
            capital_final_tradicional = df_simulado['Capital_Inversor_Tradicional'].iloc[-1]
            capital_final_ia = df_simulado['Capital_Estrategia_IA'].iloc[-1]
            
            # Calculamos cuántos días operó realmente la IA
            dias_totales = len(df_simulado)
            dias_operados = df_simulado['Señal_IA'].sum()
            porcentaje_tiempo_mercado = (dias_operados / dias_totales) * 100
            
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            col_sim1.metric("Cuenta: Tradicional", f"{capital_final_tradicional:.2f} €")
            col_sim2.metric("Cuenta: IA con Filtro", f"{capital_final_ia:.2f} €", 
                            delta=f"Dif: {(capital_final_ia - capital_final_tradicional):.2f} €")
            col_sim3.metric("Tiempo expuesto al riesgo", f"{porcentaje_tiempo_mercado:.1f}%", help="Porcentaje de días que el dinero estuvo invertido en el mercado en lugar de a salvo en efectivo.")
            
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=df_simulado.index, y=df_simulado['Capital_Inversor_Tradicional'], 
                                         name='Tradicional', line=dict(color='gray')))
            fig_sim.add_trace(go.Scatter(x=df_simulado.index, y=df_simulado['Capital_Estrategia_IA'], 
                                         name=f'Estrategia IA (> {umbral_conviccion}%)', line=dict(color='green', width=3)))
            fig_sim.update_layout(template='plotly_dark', height=300, yaxis_title='Capital en Euros (€)', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.markdown("---")
            
            # --- ZONA DEL ORÁCULO ---
            st.subheader("Señal para Mañana (Tiempo Real)")
            datos_hoy = datos_procesados.iloc[-1:]
            
            # NUEVO: Extraemos la probabilidad exacta para hoy
            probabilidades_hoy = modelo.predict_proba(datos_hoy[pistas])[0]
            probabilidad_subida = probabilidades_hoy[1] * 100
            
            precio_actual = datos_hoy['Close'].values[0]
            rsi_hoy = datos_hoy['RSI_14'].values[0]
            
            st.markdown(f"**Nivel de Convicción de la IA hoy:** {probabilidad_subida:.2f}%")
            st.progress(probabilidad_subida / 100)
            
            # El Semáforo ahora depende de tu filtro, no solo de superar el 50%
            if probabilidad_subida >= umbral_conviccion:
                st.success(f"🟢 LUZ VERDE: La IA supera tu filtro del {umbral_conviccion}%. Entorno altamente favorable.")
                st.subheader("🛡️ Instrucciones de Ejecución (Broker)")
                acciones, inversion, riesgo_max = calcular_tamaño_posicion(capital_usuario, precio_actual, stop_loss_usuario)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Acciones a Comprar", f"{acciones}")
                col2.metric("Capital a Invertir", f"{inversion:.2f} €")
                col3.metric("Riesgo Máximo", f"{riesgo_max:.2f} €")
                st.info(f"👉 **Orden:** Compra {acciones} acciones de {ticker_usuario}. Stop-Loss al -{stop_loss_usuario}%. Pérdida máxima bloqueada en {riesgo_max:.2f} €.")
            else:
                st.error(f"🔴 LUZ ROJA: Convicción insuficiente (< {umbral_conviccion}%). Mantener liquidez, ahorrar comisiones y proteger capital.")
                st.caption("Nota: El algoritmo prefiere quedarse en efectivo y no pagar al broker si no está muy seguro de la operación.")
