# ==============================================================================
# ARQUITECTURA FASE 11.2: PROTOCOLO DE TRANSPARENCIA TOTAL (V9)
# Objetivo: Forzar la identidad del Project ID y auditar la visibilidad del entorno.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

# --- CONEXIÓN SATELITAL (PROTOCOLO ROBUSTO) ---
from google.cloud import firestore as google_firestore
from firebase_admin import initialize_app, _apps

# ------------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE LA NUBE (DETECCIÓN MULTI-CAPA)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📊")

# 1.1. Buscador de Identidad (Exploración de variables de entorno)
app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_raw = os.environ.get('__firebase_config')

detected_id = None

# Intento 1: Parseo de la variable de configuración estándar
if firebase_config_raw:
    try:
        config_dict = json.loads(firebase_config_raw)
        detected_id = config_dict.get('projectId')
    except:
        pass

# Intento 2: Búsqueda en secretos de Streamlit
if not detected_id:
    try:
        detected_id = st.secrets.get("general", {}).get("project_id")
    except:
        pass

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: ESTADO DE LA RED Y DEBUG
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Estado de la Red")

# Si el sistema sigue fallando, permitimos al Arquitecto forzar el ID manualmente
project_id_manual = st.sidebar.text_input(
    "Forzar Project ID (Manual)", 
    value=detected_id if detected_id else "",
    help="Si el radar dice 'No detectado', busca el ID en tu consola Firebase y pégalo aquí."
)

db = None
if project_id_manual:
    try:
        # Forzamos la variable de entorno global para todas las librerías de Google
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id_manual
        
        # Inicializamos el cliente principal
        db = google_firestore.Client(project=project_id_manual)
        
        # Inicializamos Firebase Admin como respaldo
        if not _apps:
            initialize_app(options={'projectId': project_id_manual})
            
        st.sidebar.success(f"✅ Satélite Sintonizado: {project_id_manual}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error de Conexión: {e}")
else:
    st.sidebar.error("❌ Error: No se detectó ID de Proyecto.")
    st.sidebar.info("💡 Instrucción: Si el cuadro de arriba está vacío, escribe manualmente el ID de tu proyecto Firebase.")

# DEBUG: Solo para el Arquitecto (Ayuda a entender qué ve la IA)
with st.sidebar.expander("🔍 Debug de Identidad"):
    st.write(f"Config Detectada: {'SÍ' if firebase_config_raw else 'NO'}")
    st.write(f"App ID: {app_id}")

# ------------------------------------------------------------------------------
# 3. PANEL DE CONTROL DE INVERSIÓN
# ------------------------------------------------------------------------------
st.sidebar.header("📈 Configuración del Radar")
tickers_input = st.sidebar.text_input("Activos (ej: SPY, BTC-USD, AAPL)", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Auditar Nube")

# ------------------------------------------------------------------------------
# 4. MOTOR LÓGICO IA
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

# --- PERSISTENCIA Y AUDITORÍA ---
def guardar_en_nube(data_list):
    if db is None: return
    for item in data_list:
        try:
            # Ruta obligatoria: artifacts/{appId}/public/data/{collection}
            db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').document().set(item)
        except: pass

def realizar_auditoria_nube():
    if db is None: return []
    try:
        docs = db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').order_by('Fecha', direction=google_firestore.Query.DESCENDING).limit(20).stream()
        registros = [doc.to_dict() for doc in docs]
        if not registros: return []
        auditados = []
        for reg in registros:
            ticker = reg['Activo']
            info = yf.Ticker(ticker).history(period="2d")
            if len(info) >= 2:
                precio_pred = reg['Precio ($)']
                precio_hoy = info['Close'].iloc[-1]
                subio = 1 if precio_hoy > precio_pred else 0
                conv_alta = 1 if reg['Convicción (%)'] >= 50 else 0
                reg['Resultado'] = "✅ ACIERTO" if subio == conv_alta else "❌ ERROR"
                reg['Var %'] = round(((precio_hoy / precio_pred) - 1) * 100, 2)
                auditados.append(reg)
        return auditados
    except: return []

# ------------------------------------------------------------------------------
# 5. INTERFAZ WEB (TABS)
# ------------------------------------------------------------------------------
st.title("🤖 Portal de Inteligencia Cuantitativa")

t1, t2 = st.tabs(["🎯 Radar de Hoy", "📊 Auditoría de Aciertos"])

with t1:
    if boton_analizar:
        activos = [x.strip().upper() for x in tickers_input.split(',')]
        resultados = []
        prog = st.progress(0)
        for i, tick in enumerate(activos):
            try:
                df_raw = obtener_datos(tick)
                if not df_raw.empty:
                    df = calcular_indicadores(df_raw)
                    mod, prec, pists = entrenar_modelo(df)
                    hoy = df.iloc[-1:]
                    prob = mod.predict_proba(hoy[pists])[0][1] * 100
                    resultados.append({
                        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Activo": tick,
                        "Convicción (%)": round(prob, 2),
                        "Precio ($)": round(hoy['Close'].values[0], 2),
                        "Precisión (%)": round(prec, 2)
                    })
            except: pass
            prog.progress((i + 1) / len(activos))

        if resultados:
            if db: guardar_en_nube(resultados)
            df_res = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
            st.subheader("🏆 Ranking Institucional")
            # Restauración del color verde solicitado
            st.dataframe(df_res.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), use_container_width=True)
            
            ganador = df_res.iloc[0]
            if ganador["Convicción (%)"] >= umbral_conviccion:
                st.success(f"👑 ELEGIDO: {ganador['Activo']} ({ganador['Convicción (%)']}%)")
                acc, inv, ries = calcular_ejecucion_fraccionada(capital_usuario, ganador["Precio ($)"], stop_loss_usuario)
                c1, c2, c3 = st.columns(3)
                c1.metric("Acciones (Fracciones)", f"{acc:.4f}")
                c2.metric("Inversión", f"{inv:.2f} €")
                c3.metric("Riesgo Máximo", f"{ries:.2f} €")
                st.info(f"👉 **Orden:** Compra {acc:.4f} de {ganador['Activo']}.")
            else:
                st.error("Ningún activo superó el filtro. Liquidez al 100%.")

with t2:
    st.subheader("🌐 Marcador del Estadio (Realidad vs IA)")
    if db:
        auditoria = realizar_auditoria_nube()
        if auditoria:
            df_aud = pd.DataFrame(auditoria)
            aciertos = len(df_aud[df_aud['Resultado'] == "✅ ACIERTO"])
            rate = (aciertos / len(df_aud)) * 100
            ca1, ca2 = st.columns(2)
            ca1.metric("Hit Rate Real", f"{rate:.2f}%", delta="vs Mercado")
            ca2.metric("Muestras", len(df_aud))
            st.table(df_aud[['Fecha', 'Activo', 'Convicción (%)', 'Var %', 'Resultado']])
        else:
            st.info("Analiza activos para empezar a generar el historial de aciertos.")
    else:
        st.warning("⚠️ Conecta el satélite (Project ID) para habilitar la auditoría.")
