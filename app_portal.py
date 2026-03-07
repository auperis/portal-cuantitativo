# ==============================================================================
# ARQUITECTURA FASE 11 (CORREGIDA V4): PERSISTENCIA TOTAL + UI PREMIUM
# Objetivo: Forzar el Project ID directamente en el cliente de Firestore.
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
import json
import os

# --- CONEXIÓN SATELITAL (FIRESTORE) ---
from firebase_admin import firestore, initialize_app, credentials, _apps
import firebase_admin

# ------------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE LA NUBE (SOLUCIÓN DEFINITIVA POR INYECCIÓN DIRECTA)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📡")

# Extraemos la configuración del entorno proporcionada por el sistema
app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_str = os.environ.get('__firebase_config')

# PARSEO DE SEGURIDAD
project_id = None
if firebase_config_str:
    try:
        config_dict = json.loads(firebase_config_str)
        project_id = config_dict.get('projectId')
    except Exception as e:
        st.sidebar.error(f"Error al leer config: {e}")

# Inicializamos la App y el Cliente con inyección directa del Project ID
try:
    if not _apps:
        if project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            initialize_app(options={'projectId': project_id})
        else:
            initialize_app()
    
    # CAMBIO CRÍTICO: Pasamos el project_id directamente al cliente de firestore
    # Esto soluciona el error donde initialize_app no propaga el ID al cliente.
    db = firestore.client(project=project_id)
except Exception as e:
    # Si falla la conexión, mostramos un aviso pero no bloqueamos la web
    st.error(f"Error al conectar con Firestore: {e}")
    db = None

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL (Barra Lateral)
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Configuración del Radar")
tickers_input = st.sidebar.text_input("Lista de Activos", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Guardar en la Nube")

# ------------------------------------------------------------------------------
# 3. MOTOR LÓGICO
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    return yf.Ticker(ticker).history(period="3y")

def calcular_indicadores(df):
    datos = df.copy()
    datos['Retorno_Hoy_%'] = datos['Close'].pct_change() * 100
    datos['Volatilidad_5D'] = datos['Close'].rolling(window=5).std()
    datos['Media_Volumen_20D'] = datos['Volume'].rolling(window=20).mean()
    datos['Volumen_Relativo'] = datos['Volume'] / datos['Media_Volumen_20D']
    datos['Retorno_3D_%'] = datos['Close'].pct_change(periods=3) * 100
    
    delta = datos['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    ema_down = ema_down.replace(0, 0.001)
    datos['RSI_14'] = 100 - (100 / (1 + (ema_up / ema_down)))
    
    datos['Target_Mañana_Sube'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)
    return datos.dropna()

def entrenar_modelo(df):
    pistas = ['Retorno_Hoy_%', 'Volatilidad_5D', 'Volumen_Relativo', 'Retorno_3D_%', 'RSI_14']
    corte = int(len(df) * 0.8)
    estudio, examen = df.iloc[:corte], df.iloc[corte:]
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(estudio[pistas], estudio['Target_Mañana_Sube'])
    prob = modelo.predict_proba(examen[pistas])[:, 1] * 100
    precision = accuracy_score(examen['Target_Mañana_Sube'], np.where(prob >= 50, 1, 0)) * 100
    return modelo, precision, pistas

def calcular_ejecucion_fraccionada(capital, precio, stop_loss, riesgo_max=2.0):
    perdida_max_euros = capital * (riesgo_max / 100)
    riesgo_por_accion = precio * (stop_loss / 100)
    if riesgo_por_accion <= 0: return 0, 0, 0
    acc = perdida_max_euros / riesgo_por_accion
    if (acc * precio) > capital: acc = capital / precio
    return acc, (acc * precio), perdida_max_euros

# --- PERSISTENCIA (FIRESTORE) ---
def guardar_en_nube(data_list):
    if db is None: return
    for item in data_list:
        try:
            db.collection('artifacts', app_id, 'public', 'data', 'predicciones').document().set(item)
        except: pass

def cargar_de_nube():
    if db is None: return []
    try:
        docs = db.collection('artifacts', app_id, 'public', 'data', 'predicciones').order_by('Fecha', direction=firestore.Query.DESCENDING).limit(10).stream()
        return [doc.to_dict() for doc in docs]
    except: return []

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB
# ------------------------------------------------------------------------------
st.title("🤖 Portal de Inteligencia Cuantitativa")

if boton_analizar:
    lista_activos = [x.strip().upper() for x in tickers_input.split(',')]
    resultados_hoy = []
    
    progreso = st.progress(0)
    for i, ticker in enumerate(lista_activos):
        try:
            df_raw = obtener_datos(ticker)
            if not df_raw.empty:
                df = calcular_indicadores(df_raw)
                mod, prec, pists = entrenar_modelo(df)
                hoy = df.iloc[-1:]
                prob = mod.predict_proba(hoy[pists])[0][1] * 100
                precio = hoy['Close'].values[0]
                
                resultados_hoy.append({
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Activo": ticker,
                    "Convicción (%)": round(prob, 2),
                    "Precio ($)": round(precio, 2),
                    "Precisión (%)": round(prec, 2)
                })
        except: pass
        progreso.progress((i + 1) / len(lista_activos))

    if resultados_hoy:
        guardar_en_nube(resultados_hoy)
        df_rank = pd.DataFrame(resultados_hoy).sort_values("Convicción (%)", ascending=False)
        
        st.subheader("🏆 Ranking Institucional de Hoy")
        st.dataframe(
            df_rank.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), 
            use_container_width=True
        )
        
        ganador = df_rank.iloc[0]
        if ganador["Convicción (%)"] >= umbral_conviccion:
            st.success(f"👑 ACTIVO ELEGIDO: {ganador['Activo']} ({ganador['Convicción (%)']}%)")
            acc, inv, riesgo = calcular_ejecucion_fraccionada(capital_usuario, ganador["Precio ($)"], stop_loss_usuario)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Acciones (Fracciones)", f"{acc:.4f}")
            c2.metric("Inversión Sugerida", f"{inv:.2f} €")
            c3.metric("Riesgo Máximo", f"{riesgo:.2f} €")
            st.info(f"👉 **Orden:** Compra {acc:.4f} de {ganador['Activo']}.")
            st.toast("✅ Datos sincronizados con la nube.")
        else:
            st.error("Convicción insuficiente. Mantenemos liquidez.")

st.markdown("---")
st.subheader("🌐 Registro Histórico Permanente (Cloud Storage)")
historial_nube = cargar_de_nube()

if historial_nube:
    df_cloud = pd.DataFrame(historial_nube)
    st.table(df_cloud.head(10))
else:
    st.caption("No se detectan registros históricos en la nube. Realiza un análisis para iniciar la grabación.")
