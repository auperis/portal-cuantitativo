# ==============================================================================
# ARQUITECTURA FASE 9: EL RADAR MULTI-ACTIVO (SCREENER)
# Objetivo: Escanear múltiples activos simultáneamente para evitar la inactividad.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📡")

st.title("📡 Radar Cuantitativo IA (Screener)")
st.markdown("Escáner multi-activo para localizar la mayor probabilidad estadística del mercado.")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL (Barra Lateral)
# ------------------------------------------------------------------------------
st.sidebar.header("Parámetros de Escaneo")
# NUEVO: Ahora pedimos una lista de activos separados por comas
tickers_input = st.sidebar.text_input("Activos a escanear (separados por coma)", value="SPY, GLD, QQQ, TLT")

st.sidebar.header("🛡️ Gestión de Riesgo y Convicción")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, max_value=10000, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", min_value=50, max_value=80, value=58, step=1)

boton_analizar = st.sidebar.button("Activar Radar IA")

# ------------------------------------------------------------------------------
# 3. EL MOTOR OCULTO Y EL ESCUDO
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    activo = yf.Ticker(ticker)
    return activo.history(period="3y")

def calcular_indicadores(df):
    datos = df.copy()
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    datos['Volatilidad_5D'] = datos['Close'].rolling(window=5).std()
    datos['Media_Volumen_20D'] = datos['Volume'].rolling(window=20).mean()
    datos['Volumen_Relativo'] = datos['Volume'] / datos['Media_Volumen_20D']
    datos['Retorno_3D_%'] = datos['Close'].pct_change(periods=3) * 100
    
    delta = datos['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    datos['RSI_14'] = 100 - (100 / (1 + rs))
    
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    return datos.dropna()

def entrenar_modelo(df):
    columnas_pistas = ['Retorno_Hoy_%', 'Volatilidad_5D', 'Volumen_Relativo', 'Retorno_3D_%', 'RSI_14']
    indice_corte = int(len(df) * 0.8)
    datos_estudio = df.iloc[:indice_corte]
    datos_examen = df.iloc[indice_corte:]
    
    modelo_ia = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_ia.fit(datos_estudio[columnas_pistas], datos_estudio['Target_Mañana_Sube'])
    
    probabilidades_examen = modelo_ia.predict_proba(datos_examen[columnas_pistas])
    prob_subir_array = probabilidades_examen[:, 1] * 100
    
    predicciones_base = np.where(prob_subir_array >= 50.0, 1, 0)
    precision = accuracy_score(datos_examen['Target_Mañana_Sube'], predicciones_base) * 100
    
    return modelo_ia, precision, columnas_pistas

def calcular_tamaño_posicion(capital_total, precio_accion, stop_loss_porcentaje, riesgo_maximo_porcentaje=2.0):
    riesgo_en_euros = capital_total * (riesgo_maximo_porcentaje / 100)
    riesgo_por_accion = precio_accion * (stop_loss_porcentaje / 100)
    if riesgo_por_accion <= 0: return 0, 0, 0
    numero_acciones = int(riesgo_en_euros / riesgo_por_accion)
    capital_a_invertir = numero_acciones * precio_accion
    return numero_acciones, capital_a_invertir, riesgo_en_euros

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB (EL BUCLE DE ESCANEO)
# ------------------------------------------------------------------------------
if boton_analizar:
    # Limpiamos la lista de activos que introdujo el usuario
    lista_activos = [x.strip().upper() for x in tickers_input.split(',')]
    resultados_escaneo = []
    
    st.markdown(f"### 📡 Escaneando {len(lista_activos)} activos simultáneamente...")
    barra_progreso = st.progress(0)
    
    # BUCLE PRINCIPAL: Enviamos al ojeador a cada activo uno por uno
    for idx, ticker in enumerate(lista_activos):
        with st.spinner(f"Entrenando red neuronal para {ticker}..."):
            try:
                datos_crudos = obtener_datos(ticker)
                if not datos_crudos.empty and len(datos_crudos) > 50:
                    datos_procesados = calcular_indicadores(datos_crudos)
                    modelo, precision_ia, pistas = entrenar_modelo(datos_procesados)
                    
                    # Extraemos los datos de HOY para este activo
                    datos_hoy = datos_procesados.iloc[-1:]
                    probabilidades_hoy = modelo.predict_proba(datos_hoy[pistas])[0]
                    probabilidad_subida = probabilidades_hoy[1] * 100
                    precio_actual = datos_hoy['Close'].values[0]
                    rsi_hoy = datos_hoy['RSI_14'].values[0]
                    
                    # Guardamos el informe en nuestra tabla de resultados
                    resultados_escaneo.append({
                        "Activo": ticker,
                        "Convicción de Subida (%)": round(probabilidad_subida, 2),
                        "RSI Actual": round(rsi_hoy, 2),
                        "Precisión Histórica (%)": round(precision_ia, 2),
                        "Precio ($)": round(precio_actual, 2)
                    })
            except Exception as e:
                st.warning(f"No se pudo procesar {ticker}. Posible error de símbolo.")
                
        # Actualizamos la barra de progreso visual
        barra_progreso.progress((idx + 1) / len(lista_activos))
    
    # --- ZONA DE RANKING VISUAL ---
    if resultados_escaneo:
        # Convertimos la lista de resultados en una tabla y la ordenamos de mayor a menor convicción
        df_ranking = pd.DataFrame(resultados_escaneo)
        df_ranking = df_ranking.sort_values(by="Convicción de Subida (%)", ascending=False).reset_index(drop=True)
        
        st.markdown("---")
        st.subheader("🏆 Ranking Institucional de Oportunidades")
        
        # Mostramos la tabla elegante en la web
        st.dataframe(df_ranking.style.background_gradient(cmap='Greens', subset=['Convicción de Subida (%)']), use_container_width=True)
        
        # --- EL GANADOR ABSOLUTO ---
        mejor_activo = df_ranking.iloc[0]
        st.markdown("---")
        
        if mejor_activo["Convicción de Subida (%)"] >= umbral_conviccion:
            st.success(f"👑 ACTIVO ELEGIDO PARA DESPLIEGUE: **{mejor_activo['Activo']}**")
            st.markdown(f"La IA determina que **{mejor_activo['Activo']}** es la mejor opción matemática hoy, con un **{mejor_activo['Convicción de Subida (%)']}%** de convicción (superando tu filtro del {umbral_conviccion}%).")
            
            st.subheader("🛡️ Órdenes de Ejecución para tu Cuenta de 1.000 €")
            acciones, inversion, riesgo_max = calcular_tamaño_posicion(capital_usuario, mejor_activo["Precio ($)"], stop_loss_usuario)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Acciones a Comprar", f"{acciones}")
            col2.metric("Capital a Invertir", f"{inversion:.2f} €")
            col3.metric("Riesgo Máximo", f"{riesgo_max:.2f} €")
            
            st.info(f"👉 **Instrucción al Broker:** Ignora el resto de la lista. Compra {acciones} acciones de {mejor_activo['Activo']}. Coloca el Stop-Loss automático al -{stop_loss_usuario}%. Si la IA se equivoca, tu pérdida está contenida en {riesgo_max:.2f} €.")
        else:
            st.error(f"🛑 EL GANADOR NO SUPERA EL FILTRO")
            st.markdown(f"El activo con mayor puntuación fue **{mejor_activo['Activo']}** (Convicción: {mejor_activo['Convicción de Subida (%)']}%). Sin embargo, **NO SUPERA** tu exigencia mínima del {umbral_conviccion}%.")
            st.warning("👉 **Instrucción de Sistema:** Ningún activo del radar es digno de arriesgar tu capital hoy. MANTENER EL 100% EN EFECTIVO (Liquidez). El coste de inactividad es menor que el coste de operar sin ventaja matemática.")
    else:
        st.error("El radar no encontró datos válidos.")
