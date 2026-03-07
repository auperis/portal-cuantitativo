# ==============================================================================
# ARQUITECTURA FASE 12.1: REFUERZO DE IDENTIDAD (FIX PROJECT ID)
# Objetivo: Garantizar la conexión con Firestore incluso si el entorno es ruidoso.
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

# --- CONEXIÓN SATELITAL (DIRECTA A GOOGLE CLOUD) ---
from google.cloud import firestore as google_firestore
from firebase_admin import initialize_app, _apps

# ------------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE LA NUBE (SISTEMA DE TRIPLE BÚSQUEDA)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📊")

# Intentamos detectar el ID automáticamente
app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_raw = os.environ.get('__firebase_config')

detected_project_id = None
if firebase_config_raw:
    try:
        config_dict = json.loads(firebase_config_raw)
        detected_project_id = config_dict.get('projectId')
    except:
        pass

# BARRA LATERAL: Permitimos la entrada manual si falla la detección automática
st.sidebar.header("📡 Estado de la Red")
project_id = st.sidebar.text_input(
    "Project ID (Identificador Satelital)", 
    value=detected_project_id if detected_project_id else "",
    help="Si no se detecta automáticamente, cópialo de la configuración de tu base de datos."
)

db = None
if project_id:
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        db = google_firestore.Client(project=project_id)
        if not _apps:
            initialize_app(options={'projectId': project_id})
        st.sidebar.success(f"📡 Satélite Sintonizado: {project_id}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error de Conexión: {e}")
else:
    st.sidebar.error("❌ Error: No se detectó ID de Proyecto.")
    st.sidebar.info("💡 Introduce el ID manualmente arriba para activar la nube.")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL
# ------------------------------------------------------------------------------
st.sidebar.header("📈 Configuración del Radar")
tickers_input = st.sidebar.text_input("Activos (separados por coma)", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Auditar Nube")

# ------------------------------------------------------------------------------
# 3. MOTOR LÓGICO E INDICADORES
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
            info_mercado = yf.Ticker(ticker).history(period="2d")
            if len(info_mercado) >= 2:
                precio_prediccion = reg['Precio ($)']
                precio_real_hoy = info_mercado['Close'].iloc[-1]
                
                subio_realmente = 1 if precio_real_hoy > precio_prediccion else 0
                conviccion_alta = 1 if reg['Convicción (%)'] >= 50 else 0
                
                reg['Resultado'] = "✅ ACIERTO" if subio_realmente == conviccion_alta else "❌ ERROR"
                reg['Variación Real (%)'] = round(((precio_real_hoy / precio_prediccion) - 1) * 100, 2)
                auditados.append(reg)
        return auditados
    except:
        return []

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN WEB
# ------------------------------------------------------------------------------
st.title("🤖 Portal de Inteligencia Cuantitativa")

tab1, tab2 = st.tabs(["🎯 Radar de Hoy", "📊 Auditoría de la IA"])

with tab1:
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
            if db: guardar_en_nube(resultados_hoy)
            df_rank = pd.DataFrame(resultados_hoy).sort_values("Convicción (%)", ascending=False)
            
            st.subheader("🏆 Ranking Institucional")
            st.dataframe(df_rank.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), use_container_width=True)
            
            ganador = df_rank.iloc[0]
            if ganador["Convicción (%)"] >= umbral_conviccion:
                st.success(f"👑 ELEGIDO: {ganador['Activo']} ({ganador['Convicción (%)']}%)")
                acc, inv, riesgo = calcular_ejecucion_fraccionada(capital_usuario, ganador["Precio ($)"], stop_loss_usuario)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Acciones (Fracciones)", f"{acc:.4f}")
                c2.metric("Inversión", f"{inv:.2f} €")
                c3.metric("Riesgo Máximo", f"{riesgo:.2f} €")
                st.info(f"👉 **Orden:** Compra {acc:.4f} de {ganador['Activo']}.")
                st.toast("✅ Datos guardados en la nube.")
            else:
                st.error("Ningún activo superó el filtro. Liquidez al 100%.")

with tab2:
    st.subheader("🌐 Marcador del Estadio (Realidad vs IA)")
    st.markdown("Comparativa de las últimas predicciones guardadas contra el precio actual del mercado.")
    
    if db:
        datos_auditados = realizar_auditoria_nube()
        if datos_auditados:
            df_auditoria = pd.DataFrame(datos_auditados)
            total = len(df_auditoria)
            aciertos = len(df_auditoria[df_auditoria['Resultado'] == "✅ ACIERTO"])
            hit_rate = (aciertos / total) * 100
            
            col_aud1, col_aud2 = st.columns(2)
            col_aud1.metric("Hit Rate Real (Caja Negra)", f"{hit_rate:.2f}%", 
                           delta="Acierto vs Mercado", delta_color="normal" if hit_rate > 50 else "inverse")
            col_aud2.metric("Muestras Auditadas", total)
            
            st.table(df_auditoria[['Fecha', 'Activo', 'Convicción (%)', 'Variación Real (%)', 'Resultado']])
        else:
            st.info("No hay datos suficientes para auditar. Ejecuta el radar para generar registros.")
    else:
        st.warning("⚠️ Conecta la base de datos (Project ID) para habilitar la auditoría.")
