# ==============================================================================
# ARQUITECTURA FASE 13: PORTAL INSTITUCIONAL INTEGRADO (SMART REBALANCING)
# Objetivo: Radar IA + Ejecución Fraccionada + Memoria Cloud + Auditoría + Pesos
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import json
import os

# --- CONEXIÓN SATELITAL (PROTOCOLO ROBUSTO V8) ---
try:
    from google.cloud import firestore as google_firestore
    from firebase_admin import initialize_app, _apps
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN E INICIALIZACIÓN DE LA NUBE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="⚖️")

# Detección de identidad automática (Inyectada por el entorno)
app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_raw = os.environ.get('__firebase_config')

detected_id = None
if firebase_config_raw:
    try:
        detected_id = json.loads(firebase_config_raw).get('projectId')
    except: pass

# --- BARRA LATERAL: ESTADO DE LA RED ---
st.sidebar.header("📡 Estado de la Red")
project_id_manual = st.sidebar.text_input(
    "Project ID (Manual)", 
    value=detected_id if detected_id else "",
    help="Pega aquí el ID de tu proyecto Firebase si el radar no lo detecta solo."
)

db = None
if HAS_FIREBASE and project_id_manual and len(project_id_manual) > 4:
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id_manual
        db = google_firestore.Client(project=project_id_manual)
        if not _apps:
            initialize_app(options={'projectId': project_id_manual})
        st.sidebar.success(f"✅ Satélite Sintonizado")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error de Enlace: {e}")
else:
    st.sidebar.warning("🏠 Modo Local Activo (Sin Nube)")

# ------------------------------------------------------------------------------
# 2. PANEL DE CONTROL DE RIESGO (Cartera 1.000 €)
# ------------------------------------------------------------------------------
st.sidebar.header("📈 Configuración del Radar")
tickers_input = st.sidebar.text_input("Activos a vigilar", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_total = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Smart Rebalancing")

# ------------------------------------------------------------------------------
# 3. MOTOR LÓGICO IA
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    return yf.Ticker(ticker).history(period="3y")

def calcular_indicadores(df):
    d = df.copy()
    d['Retorno_Hoy'] = d['Close'].pct_change() * 100
    d['Volatilidad_5D'] = d['Close'].rolling(5).std()
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Distancia_Media'] = ((d['Close'] / d['Media_20']) - 1) * 100
    # Objetivo: ¿Mañana cierra más alto que hoy?
    d['Target'] = np.where(d['Close'].shift(-1) > d['Close'], 1, 0)
    return d.dropna()

def entrenar_ia(df):
    pistas = ['Retorno_Hoy', 'Volatilidad_5D', 'Distancia_Media']
    corte = int(len(df) * 0.8)
    train = df.iloc[:corte]
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(train[pistas], train['Target'])
    return modelo, pistas

# --- GESTIÓN DE MEMORIA Y AUDITORÍA ---
def guardar_en_nube(datos):
    if db:
        try:
            db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').document().set(datos)
        except: pass

def obtener_hit_rate_real():
    if not db: return 0.52 # Valor base si no hay nube
    try:
        docs = db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').order_by('Fecha', direction=google_firestore.Query.DESCENDING).limit(10).stream()
        registros = [doc.to_dict() for doc in docs]
        if not registros: return 0.52
        
        aciertos = 0
        for reg in registros:
            info = yf.Ticker(reg['Activo']).history(period="2d")
            if len(info) >= 2:
                subio_real = 1 if info['Close'].iloc[-1] > reg['Precio ($)'] else 0
                if subio_real == (1 if reg['Convicción (%)'] >= 50 else 0):
                    aciertos += 1
        return aciertos / len(registros)
    except: return 0.52

# ------------------------------------------------------------------------------
# 4. INTERFAZ VISUAL (DASHBOARD)
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Gestión Cuantitativa de Capital")

t1, t2 = st.tabs(["🎯 Radar y Rebalanceo", "📊 Auditoría de la IA"])

with t1:
    if boton_analizar:
        activos = [x.strip().upper() for x in tickers_input.split(',')]
        resultados = []
        prog = st.progress(0)
        
        for i, tick in enumerate(activos):
            try:
                df = calcular_indicadores(obtener_datos(tick))
                mod, pists = entrenar_ia(df)
                hoy = df.iloc[-1:]
                prob = mod.predict_proba(hoy[pists])[0][1] * 100
                
                res = {
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Activo": tick,
                    "Convicción (%)": round(prob, 2),
                    "Precio ($)": round(hoy['Close'].values[0], 2)
                }
                resultados.append(res)
                guardar_en_nube(res)
            except: pass
            prog.progress((i + 1) / len(activos))

        if resultados:
            df_res = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
            
            # --- SECCIÓN DE REBALANCEO INTELIGENTE ---
            st.subheader("⚖️ Propuesta de Smart Rebalancing")
            
            ganadores = df_res[df_res["Convicción (%)"] >= umbral_conviccion]
            
            if not ganadores.empty:
                hr = obtener_hit_rate_real()
                st.info(f"Auditoría activa: Hit Rate reciente del {hr*100:.1f}%.")
                
                # Freno de seguridad: si fallamos mucho, arriesgamos menos
                riesgo_total_permitido = 0.02 if hr >= 0.45 else 0.01 # 2% o 1%
                capital_riesgo_eur = capital_total * riesgo_total_permitido
                
                # Reparto proporcional (Pesos)
                suma_conv = ganadores["Convicción (%)"].sum()
                ganadores["Peso (%)"] = (ganadores["Convicción (%)"] / suma_conv) * 100
                
                # Cálculo de Inversión y Fracciones
                # Inversión = (Capital Riesgo * Peso) / (Stop Loss %)
                ganadores["Inversión (€)"] = (ganadores["Peso (%)"] / 100) * (capital_riesgo_eur / (stop_loss_usuario / 100))
                ganadores["Acciones"] = ganadores["Inversión (€)"] / ganadores["Precio ($)"]
                
                st.dataframe(
                    ganadores[["Activo", "Convicción (%)", "Peso (%)", "Inversión (€)", "Acciones"]].style.background_gradient(cmap='Greens'), 
                    use_container_width=True
                )
                
                total_invertido = ganadores["Inversión (€)"].sum()
                st.success(f"🚀 Instrucción: Distribuir {total_invertido:.2f} € entre {len(ganadores)} activos.")
                st.caption(f"Riesgo máximo de esta cartera: {capital_riesgo_eur:.2f} € (Capital Protegido).")
            else:
                st.error(f"Ningún activo supera el {umbral_conviccion}%. Se recomienda 100% liquidez.")

with t2:
    st.subheader("🌐 Marcador de Aciertos (Caja Negra)")
    if db:
        docs = db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').order_by('Fecha', direction=google_firestore.Query.DESCENDING).limit(10).stream()
        registros = [d.to_dict() for d in docs]
        if registros:
            st.table(pd.DataFrame(registros))
        else:
            st.info("Analiza activos para empezar a grabar el historial.")
    else:
        st.warning("⚠️ Conecta el Project ID para ver el historial permanente.")
