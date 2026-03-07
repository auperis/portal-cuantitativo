# ==============================================================================
# ARQUITECTURA FASE 14: AGENTE DE NOTIFICACIONES (TELEGRAM BOT)
# Objetivo: Enviar alertas en tiempo real al móvil cuando la IA detecta señales.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import json
import os

# --- CONEXIÓN SATELITAL (PROTOCOLO HÍBRIDO) ---
try:
    from google.cloud import firestore as google_firestore
    from firebase_admin import initialize_app, _apps
    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal Cuantitativo IA", layout="wide", page_icon="📡")

# Detección de identidad automática
app_id = os.environ.get('__app_id', 'mi-portal-ia-1000')
firebase_config_raw = os.environ.get('__firebase_config')
detected_id = None
if firebase_config_raw:
    try:
        config_dict = json.loads(firebase_config_raw)
        detected_id = config_dict.get('projectId')
    except: pass

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: CENTRO DE COMUNICACIONES
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Conexión y Alertas")

# Configuración de Nube (Opcional)
project_id_manual = st.sidebar.text_input("Project ID (Nube)", value=detected_id if detected_id else "")
db = None
if HAS_CLOUD and project_id_manual:
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id_manual
        db = google_firestore.Client(project=project_id_manual)
        if not _apps: initialize_app(options={'projectId': project_id_manual})
        st.sidebar.success("✅ Nube Conectada")
    except: st.sidebar.warning("🏠 Modo Local Activo")

st.sidebar.divider()

# CONFIGURACIÓN DEL MENSAJERO (TELEGRAM) - ¡ESTO ES LO QUE BUSCABA!
st.sidebar.subheader("🤖 Configurar Bot Telegram")
tel_token = st.sidebar.text_input("Bot Token", type="password", help="Pega aquí el Token de @BotFather")
tel_chat_id = st.sidebar.text_input("Chat ID", help="Pega aquí tu ID de @userinfobot")
activar_alertas = st.sidebar.checkbox("Activar Alertas al Móvil", value=False)

# ------------------------------------------------------------------------------
# 3. MOTOR DE NOTIFICACIONES
# ------------------------------------------------------------------------------
def enviar_alerta_telegram(mensaje):
    if not tel_token or not tel_chat_id:
        return
    url = f"https://api.telegram.org/bot{tel_token}/sendMessage"
    payload = {"chat_id": tel_chat_id, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Error al enviar alerta: {e}")

# ------------------------------------------------------------------------------
# 4. MOTOR LÓGICO IA Y RIESGO
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    return yf.Ticker(ticker).history(period="2y")

def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Distancia'] = ((d['Close'] / d['Media_20']) - 1) * 100
    d['Target'] = np.where(d['Close'].shift(-1) > d['Close'], 1, 0)
    return d.dropna()

def entrenar_ia(df):
    feat = ['Retorno', 'Distancia']
    train = df.iloc[:int(len(df)*0.8)]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(train[feat], train['Target'])
    return modelo, feat

# ------------------------------------------------------------------------------
# 5. INTERFAZ Y REBALANCEO
# ------------------------------------------------------------------------------
st.title("🤖 Portal IA: Gestión Cuantitativa")

st.sidebar.header("🛡️ Gestión de Riesgo")
capital_total = st.sidebar.number_input("Capital (€)", value=1000)
umbral_conv = st.sidebar.slider("Umbral Convicción (%)", 50, 80, 65)

if st.sidebar.button("Activar Radar y Notificar"):
    tickers = ["SPY", "QQQ", "GLD", "BTC-USD", "TLT"]
    resultados = []
    
    progreso = st.progress(0)
    for i, tick in enumerate(tickers):
        try:
            df = calcular_indicadores(obtener_datos(tick))
            mod, feat = entrenar_ia(df)
            prob = mod.predict_proba(df.iloc[-1:][feat])[0][1] * 100
            precio = df['Close'].iloc[-1]
            resultados.append({
                "Activo": tick,
                "Convicción (%)": round(prob, 2),
                "Precio ($)": round(precio, 2)
            })
        except: pass
        progreso.progress((i + 1) / len(tickers))
    
    if resultados:
        df_res = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
        st.subheader("🏆 Ranking de Hoy")
        st.dataframe(df_res.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), use_container_width=True)
        
        ganador = df_res.iloc[0]
        if ganador["Convicción (%)"] >= umbral_conv:
            st.success(f"👑 SEÑAL GANADORA: {ganador['Activo']} ({ganador['Convicción (%)']}%)")
            
            # Cálculo de fracciones para 1.000€
            riesgo_eur = capital_total * 0.02 # Riesgo 20€
            stop_loss_eur = ganador['Precio ($)'] * 0.05 # 5% SL
            acciones = riesgo_eur / stop_loss_eur
            
            reporte = (
                f"🚀 *ALERTA PORTAL IA*\n\n"
                f"Activo: `{ganador['Activo']}`\n"
                f"Convicción: `{ganador['Convicción (%)']}%` 📈\n"
                f"Precio: `{ganador['Precio ($)']} $`\n\n"
                f"💡 *Instrucción de Compra:*\n"
                f"Comprar `{acciones:.4f}` acciones.\n"
                f"Inversión: `{acciones * ganador['Precio ($)']:.2f} €`"
            )
            
            if activar_alertas:
                enviar_alerta_telegram(reporte)
                st.toast("📲 Alerta enviada al móvil.")
        else:
            st.error(f"Convicción insuficiente ({ganador['Convicción (%)']}%). Liquidez al 100%.")
