# ==============================================================================
# ARQUITECTURA FASE 12.5: ESTABILIZACIÓN Y MODO LOCAL
# Objetivo: Que el sistema funcione SIEMPRE, con o sin conexión a la nube.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os
import json

# Intentamos cargar la librería de la nube, pero si no está, no bloqueamos al usuario
try:
    from google.cloud import firestore as google_firestore
    from firebase_admin import initialize_app, _apps
    HAS_CLOUD_CAPABILITY = True
except ImportError:
    HAS_CLOUD_CAPABILITY = False

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA INTERFAZ
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal de Inversión IA", layout="wide", page_icon="🛡️")

# Inicializamos una memoria temporal (solo dura mientras la pestaña esté abierta)
if 'memoria_temporal' not in st.session_state:
    st.session_state['memoria_temporal'] = []

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL (EL CENTRO DE CONTROL)
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Conexión al Sistema")

# Recuperamos configuración si existe en el entorno
firebase_config_raw = os.environ.get('__firebase_config')
detected_id = None
if firebase_config_raw:
    try:
        detected_id = json.loads(firebase_config_raw).get('projectId')
    except: pass

# CUADRO DE TEXTO CLAVE: Aquí es donde pegas el ID cuando lo tengas
project_id = st.sidebar.text_input(8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA
    "Project ID (Opcional)", 
    value=detected_id if detected_id else "",
    placeholder="ej: portal-ia-12345",
    help="Si lo dejas vacío, el sistema funcionará en modo local (sin memoria histórica)."
)

db = None
MODO_NUBE = False

# Intentamos sintonizar el satélite solo si hay un ID válido
if HAS_CLOUD_CAPABILITY and project_id and len(project_id) > 5:
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        db = google_firestore.Client(project=project_id)
        if not _apps:
            initialize_app(options={'projectId': project_id})
        st.sidebar.success("✅ Modo Satélite Activo")
        MODO_NUBE = True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Usando Modo Local (Error: {e})")
else:
    st.sidebar.info("🏠 Modo Local Activo")

st.sidebar.divider()
st.sidebar.header("📈 Configuración del Radar")
activos_lista = st.sidebar.text_input("Activos (separados por coma)", value="SPY, QQQ, BTC-USD, GLD")
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 1.0, 10.0, 5.0)

boton_ejecutar = st.sidebar.button("🚀 Activar Radar IA")

# ------------------------------------------------------------------------------
# 3. MOTOR IA Y CÁLCULOS
# ------------------------------------------------------------------------------
def obtener_datos(ticker):
    return yf.Ticker(ticker).history(period="2y")

def calcular_indicadores(df):
    d = df.copy()
    d['Retorno_Hoy'] = d['Close'].pct_change() * 100
    d['Media_20'] = d['Close'].rolling(20).mean()
    d['Distancia'] = ((d['Close'] / d['Media_20']) - 1) * 100
    d['Sube_Mañana'] = np.where(d['Close'].shift(-1) > d['Close'], 1, 0)
    return d.dropna()

def entrenar_cerebro(df):
    features = ['Retorno_Hoy', 'Distancia']
    corte = int(len(df) * 0.8)
    train = df.iloc[:corte]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(train[features], train['Sube_Mañana'])
    return model, features

# ------------------------------------------------------------------------------
# 4. INTERFAZ PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🤖 Portal Cuantitativo IA (Cartera 1.000 €)")

tab1, tab2 = st.tabs(["🎯 Radar de Hoy", "📊 Historial / Auditoría"])

with tab1:
    if boton_ejecutar:
        activos = [x.strip().upper() for x in activos_lista.split(',')]
        resultados = []
        progreso = st.progress(0)
        
        for i, tick in enumerate(activos):
            try:
                # 1. Ingesta y Procesamiento
                df = calcular_indicadores(obtener_datos(tick))
                model, feat = entrenar_cerebro(df)
                
                # 2. Predicción para Mañana
                hoy = df.iloc[-1:]
                prob = model.predict_proba(hoy[feat])[0][1] * 100
                
                res = {
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Activo": tick,
                    "Convicción (%)": round(prob, 2),
                    "Precio ($)": round(hoy['Close'].values[0], 2)
                }
                resultados.append(res)
                
                # 3. Guardado (Nube o Local)
                if MODO_NUBE and db:
                    app_id_env = os.environ.get('__app_id', 'default')
                    db.collection('artifacts').document(app_id_env).collection('public').document('data').collection('predicciones').document().set(res)
                
                st.session_state['memoria_temporal'].append(res)
                
            except: pass
            progreso.progress((i + 1) / len(activos))

        if resultados:
            df_final = pd.DataFrame(resultados).sort_values("Convicción (%)", ascending=False)
            st.subheader("🏆 Ranking de Oportunidades")
            st.dataframe(df_final.style.background_gradient(cmap='Greens', subset=['Convicción (%)']), use_container_width=True)
            
            ganador = df_final.iloc[0]
            st.success(f"👑 Señal Ganadora: {ganador['Activo']} con {ganador['Convicción (%)']}% de probabilidad.")
            
            # Gestión de Riesgo (Fase 10)
            riesgo_eur = capital_total * 0.02 # Arriesgamos solo 20€
            coste_stop = ganador['Precio ($)'] * (stop_loss_usuario / 100 if 'stop_loss_usuario' in locals() else stop_loss_pct / 100)
            acciones = riesgo_eur / coste_stop
            
            col1, col2 = st.columns(2)
            col1.metric("Acciones (Fraccionadas)", f"{acciones:.4f}")
            col2.metric("Inversión Sugerida", f"{acciones * ganador['Precio ($)']:.2f} €")
            st.info(f"Riesgo Máximo de la operación: {riesgo_eur} €.")

with tab2:
    st.subheader("📓 Historial de Predicciones")
    if st.session_state['memoria_temporal']:
        df_hist = pd.DataFrame(st.session_state['memoria_temporal'])
        st.table(df_hist.tail(10))
        if not MODO_NUBE:
            st.warning("⚠️ Nota: Estás en Modo Local. Este historial se borrará si cierras la pestaña.")
    else:
        st.info("Activa el radar en la pestaña anterior para ver datos aquí.")
