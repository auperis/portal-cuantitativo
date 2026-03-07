# ==============================================================================
# ARQUITECTURA FASE 14.3: DIAGNÓSTICO DE COMUNICACIONES Y BOT DETECTOR
# Objetivo: Forzar la aparición del botón de prueba e identificar el Bot ADN.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL DEL PORTAL
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Torre de Control", layout="wide", page_icon="📡")

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: EL CEREBRO DE CONEXIONES
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Conexión de Mensajería")

# Credenciales fijas del Arquitecto (ADN del Sistema)
TOKEN_FIJO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_FIJO = "1063578190"

token_input = st.sidebar.text_input("Bot Token (ADN)", value=TOKEN_FIJO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID (Tu Dirección)", value=CHAT_ID_FIJO)

# --- DETECTOR AUTOMÁTICO DE IDENTIDAD DEL BOT ---
def detectar_bot(token):
    if not token: return None
    try:
        url = f"https://api.telegram.org/bot{token}/getMe"
        r = requests.get(url).json()
        if r.get("ok"):
            return r["result"]["first_name"], r["result"]["username"]
    except:
        pass
    return None

identidad = detectar_bot(token_input)

if identidad:
    nombre_bot, usuario_bot = identidad
    st.sidebar.success(f"🤖 Bot Identificado: **{nombre_bot}** (@{usuario_bot})")
else:
    st.sidebar.error("❌ No se detecta ningún Bot con ese Token.")

st.sidebar.divider()

# --- BOTÓN AZUL DE PRUEBA (EL DIAGNÓSTICO) ---
st.sidebar.subheader("🛠️ Diagnóstico de Enlace")
mensaje_test = "🔔 *SISTEMA IA ONLINE*\nArquitecto, este es un mensaje de prueba. Si lees esto, el puente de mando está operativo."

if st.sidebar.button("🔵 ENVIAR MENSAJE DE PRUEBA"):
    url_send = f"https://api.telegram.org/bot{token_input}/sendMessage"
    payload = {"chat_id": chat_id_input, "text": mensaje_test, "parse_mode": "Markdown"}
    try:
        res = requests.post(url_send, json=payload)
        if res.status_code == 200:
            st.sidebar.success("✅ ¡Mensaje enviado! Revisa tu Telegram.")
        else:
            st.sidebar.error(f"❌ Error de entrega (Código {res.status_code})")
            st.sidebar.info("Asegúrate de haber pulsado 'INICIAR' en el chat del Bot.")
    except Exception as e:
        st.sidebar.error(f"Fallo de conexión: {e}")

# ------------------------------------------------------------------------------
# 3. CONFIGURACIÓN DE CARTERA (1.000 €)
# ------------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header("🛡️ Gestión de Riesgo")
capital_total = st.sidebar.number_input("Capital en Gestión (€)", value=1000)
activar_alertas = st.sidebar.checkbox("Activar Alertas de Trading", value=True)

# ------------------------------------------------------------------------------
# 4. MOTOR LÓGICO IA (RADAR)
# ------------------------------------------------------------------------------
def ejecutar_radar():
    activos = ["SPY", "QQQ", "BTC-USD", "GLD"]
    resultados = []
    
    for tick in activos:
        try:
            # Ingesta de datos
            df = yf.Ticker(tick).history(period="1y")
            df['Retorno'] = df['Close'].pct_change()
            df['Sube'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            df = df.dropna()
            
            # Entrenamiento rápido
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(df[['Retorno']], df['Sube'])
            
            # Predicción
            prob = model.predict_proba(df[['Retorno']].iloc[-1:]) [0][1] * 100
            precio = df['Close'].iloc[-1]
            
            resultados.append({
                "Activo": tick,
                "Convicción (%)": round(prob, 2),
                "Precio ($)": round(precio, 2)
            })
        except:
            pass
    return resultados

# ------------------------------------------------------------------------------
# 5. PANEL DE CONTROL PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🤖 Portal Cuantitativo IA")
st.markdown("### Auditoría de Comunicaciones e Inversión")

if st.button("🚀 Iniciar Escaneo de Mercado"):
    with st.spinner("Analizando convergencias algorítmicas..."):
        data_ia = ejecutar_radar()
        
        if data_ia:
            df_final = pd.DataFrame(data_ia).sort_values("Convicción (%)", ascending=False)
            st.table(df_final)
            
            ganador = df_final.iloc[0]
            st.subheader(f"🎯 Oportunidad Detectada: {ganador['Activo']}")
            
            # Instrucción de compra (2% de riesgo)
            inv_sugerida = capital_total * 0.1 # 100€ de los 1.000€
            
            instruccion = (
                f"🚀 *NUEVA SEÑAL IA*\n\n"
                f"Activo: `{ganador['Activo']}`\n"
                f"Convicción: `{ganador['Convicción (%)']}%` 📈\n"
                f"Sugerencia: Invertir `{inv_sugerida} €`"
            )
            
            if activar_alertas:
                # Enviar alerta real
                requests.post(f"https://api.telegram.org/bot{token_input}/sendMessage", 
                             json={"chat_id": chat_id_input, "text": instruccion, "parse_mode": "Markdown"})
                st.toast("📲 Alerta enviada a Telegram")
