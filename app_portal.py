# ==============================================================================
# ARQUITECTURA FASE 13: SMART REBALANCING (EL PILOTO AUTOMÁTICO)
# Objetivo: Distribuir los 1.000 € entre los mejores activos según su convicción.
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

# --- CONEXIÓN SATELITAL ---
from google.cloud import firestore as google_firestore
from firebase_admin import initialize_app, _apps

# ------------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE LA NUBE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="🎯")

app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_raw = os.environ.get('__firebase_config')

detected_id = None
if firebase_config_raw:
    try:
        config_dict = json.loads(firebase_config_raw)
        detected_id = config_dict.get('projectId')
    except:
        pass

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: ESTADO DE LA RED
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Conexión Satelital")

project_id_manual = st.sidebar.text_input(
    "Introducir Project ID Real", 
    value=detected_id if detected_id else "",
    placeholder="ej: portal-ia-33421"
)

db = None
if project_id_manual and len(project_id_manual) > 4:
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id_manual
        db = google_firestore.Client(project=project_id_manual)
        if not _apps:
            initialize_app(options={'projectId': project_id_manual})
        st.sidebar.success(f"✅ Satélite Sintonizado")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")

# ------------------------------------------------------------------------------
# 3. PANEL DE CONTROL DE RIESGO
# ------------------------------------------------------------------------------
st.sidebar.header("📈 Configuración del Radar")
tickers_input = st.sidebar.text_input("Activos a vigilar", value="SPY, GLD, QQQ, TLT, BTC-USD")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_usuario = st.sidebar.number_input("Capital Total (€)", min_value=100, value=1000)
stop_loss_usuario = st.sidebar.slider("Stop-Loss (Paracaídas %)", 1.0, 15.0, 5.0, 0.5)
umbral_conviccion = st.sidebar.slider("Filtro de Convicción IA (%)", 50, 80, 58)

boton_analizar = st.sidebar.button("Activar Radar y Smart Rebalancing")

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

def entrenar_cerebro_ia(df):
    pistas = ['Retorno_Hoy_%', 'Volatilidad_5D', 'Volumen_Relativo', 'Retorno_3D_%', 'RSI_14']
    corte = int(len(df) * 0.8)
    estudio = df.iloc[:corte]
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(estudio[pistas], estudio['Target_Mañana_Sube'])
    return modelo, pistas

# --- PERSISTENCIA Y AUDITORÍA ---
def guardar_en_nube(data_list):
    if db is None: return
    for item in data_list:
        try:
            db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').document().set(item)
        except: pass

def obtener_hit_rate_reciente():
    if db is None: return 0.5 # 50% por defecto si no hay datos
    try:
        docs = db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').order_by('Fecha', direction=google_firestore.Query.DESCENDING).limit(10).stream()
        registros = [doc.to_dict() for doc in docs]
        if not registros: return 0.5
        
        aciertos = 0
        for reg in registros:
            info = yf.Ticker(reg['Activo']).history(period="2d")
            if len(info) >= 2:
                subio = 1 if info['Close'].iloc[-1] > reg['Precio ($)'] else 0
                conv_alta = 1 if reg['Convicción (%)'] >= 50 else 0
                if subio == conv_alta: aciertos += 1
        return aciertos / len(registros)
    except: return 0.5

# ------------------------------------------------------------------------------
# 5. INTERFAZ WEB
# ------------------------------------------------------------------------------
st.title("🤖 Portal de Inteligencia Cuantitativa")

t1, t2 = st.tabs(["🎯 Radar y Rebalanceo", "📊 Auditoría IA"])

with t1:
    if boton_analizar:
        activos = [x.strip().upper() for x in tickers_input.split(',')]
        resultados = []
        
        # 1. Ejecución del Radar
        prog = st.progress(0)
        for i, tick in enumerate(activos):
            try:
                df_raw = obtener_datos(tick)
                if not df_raw.empty:
                    df = calcular_indicadores(df_raw)
                    mod, pists = entrenar_cerebro_ia(df)
                    hoy = df.iloc[-1:]
                    prob = mod.predict_proba(hoy[pists])[0][1] * 100
                    resultados.append({
                        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Activo": tick,
                        "Convicción (%)": round(prob, 2),
                        "Precio ($)": round(hoy['Close'].values[0], 2)
                    })
            except: pass
            prog.progress((i + 1) / len(activos))

        if resultados:
            if db: guardar_en_nube(resultados)
            df_res = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
            
            # 2. LÓGICA DE SMART REBALANCING
            st.subheader("⚖️ Propuesta de Rebalanceo Inteligente")
            
            # Filtramos solo activos que superan el umbral
            ganadores = df_res[df_res["Convicción (%)"] >= umbral_conviccion]
            
            if not ganadores.empty:
                # Obtenemos el Hit Rate para ajustar el riesgo
                hr = obtener_hit_rate_reciente()
                st.info(f"Auditoría activa: El Hit Rate reciente es del {hr*100:.1f}%. Ajustando exposición...")
                
                # Si el Hit Rate es malo (<45%), reducimos la inversión a la mitad por seguridad
                multiplicador_seguridad = 1.0 if hr >= 0.45 else 0.5
                capital_disponible = capital_usuario * 0.2 * multiplicador_seguridad # Usamos max 20% del capital total
                
                # Repartimos el capital disponible proporcionalmente a la convicción
                suma_conviccion = ganadores["Convicción (%)"].sum()
                ganadores["Peso (%)"] = (ganadores["Convicción (%)"] / suma_conviccion) * 100
                ganadores["Inversión (€)"] = (ganadores["Peso (%)"] / 100) * capital_disponible
                ganadores["Acciones"] = ganadores["Inversión (€)"] / ganadores["Precio ($)"]
                
                st.dataframe(ganadores[["Activo", "Convicción (%)", "Peso (%)", "Inversión (€)", "Acciones"]].style.background_gradient(cmap='Greens'))
                
                st.success(f"🚀 Acción Recomendada: Distribuir {capital_disponible:.2f} € entre {len(ganadores)} activos.")
            else:
                st.error("Ningún activo tiene suficiente convicción. Mantener 1.000 € en liquidez.")

with t2:
    st.subheader("🌐 Auditoría de la Caja Negra")
    if db:
        docs = db.collection('artifacts').document(app_id).collection('public').document('data').collection('predicciones').order_by('Fecha', direction=google_firestore.Query.DESCENDING).limit(10).stream()
        registros = [doc.to_dict() for doc in docs]
        if registros:
            st.table(pd.DataFrame(registros))
        else:
            st.info("Analiza activos para generar historial.")
    else:
        st.warning("⚠️ Conecta el satélite para ver la auditoría.")
