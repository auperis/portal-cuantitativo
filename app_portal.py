# ==============================================================================
# ARQUITECTURA FASE 38: EL LECTOR DE BITÁCORA (LOCAL LEDGER)
# Objetivo: Permitir la ingesta local de archivos CSV para auditar y analizar
# las decisiones pasadas de la IA sin depender de bases de datos en la nube.
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os
import plotly.express as px  # NUEVO: Librería de Business Intelligence (Fase 41)

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL (MODO INSTITUCIONAL)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Integración Órdenes", layout="wide", page_icon="🏦")

if 'auto_bias' not in st.session_state:
    st.session_state['auto_bias'] = 0.0

# ------------------------------------------------------------------------------
# 2. BARRA LATERAL: AJUSTES ADN Y RIESGO
# ------------------------------------------------------------------------------
st.sidebar.header("📡 Comunicaciones (ADN)")
TOKEN_ARQUITECTO = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID_ARQUITECTO = "1063578190"

token_input = st.sidebar.text_input("Bot Token", value=TOKEN_ARQUITECTO, type="password")
chat_id_input = st.sidebar.text_input("Chat ID", value=CHAT_ID_ARQUITECTO)
activar_alertas = st.sidebar.checkbox("Activar Alertas Telegram", value=True)

st.sidebar.divider()
st.sidebar.header("🌍 Universo de Escaneo")
escanear_indices = st.sidebar.checkbox("Índices (SPY, QQQ, IWM)", value=True)
escanear_sectores = st.sidebar.checkbox("Sectores (XLK, XLF, XLV, XLE)", value=True)
escanear_refugios = st.sidebar.checkbox("Refugios (GLD, TLT, BTC-USD)", value=True)

# Hemos ocultado la caja fuerte de Firebase a petición del Arquitecto (Modo Local On-Premise)
st.sidebar.divider()
st.sidebar.header("🎛️ Calibración IA")
dias_vision_ia = st.sidebar.slider("Visión IA (Días)", 1, 15, 5)
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 70.0, 52.0)
multiplicador_atr = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0)
filtro_macro = st.sidebar.checkbox("Lente Macro (Media 200)", value=True)

st.sidebar.divider()
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)
max_exposicion = st.sidebar.slider("Exposición Máxima (%)", 5.0, 40.0, 25.0)

st.sidebar.divider()
st.sidebar.header("💰 Gestión de Liquidez (Parking)")
apy_liquidez = st.sidebar.slider("Fondo Monetario (APY %)", 0.0, 10.0, 3.5)
dias_simulacion = st.sidebar.selectbox("Días de Simulación Auditoría", [90, 180, 365], index=1)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# ------------------------------------------------------------------------------

# NUEVO: FASE 40 - El Bibliotecario Inteligente (Memoria Caché)
# @st.cache_data le dice a Streamlit que guarde el resultado de esta función.
# ttl=3600 significa "Time To Live": la fotocopia caduca y se borra a los 3600 segundos (1 hora).
@st.cache_data(ttl=3600)
def descargar_datos_cacheados(ticker, periodo="3y"):
    """Descarga datos de Yahoo y los memoriza para no saturar la API."""
    return yf.Ticker(ticker).history(period=periodo)

def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_50'] = d['Close'].rolling(50).mean()
    d['Media_200'] = d['Close'].rolling(200).mean() 
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    d['High_Low'] = d['High'] - d['Low']
    d['High_PrevClose'] = np.abs(d['High'] - d['Close'].shift(1))
    d['Low_PrevClose'] = np.abs(d['Low'] - d['Close'].shift(1))
    d['True_Range'] = d[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    d['ATR'] = d['True_Range'].rolling(14).mean()
    d['ATR_pct'] = (d['ATR'] / d['Close']) * 100
    return d.dropna()

def entrenar_ia_radar(ticker, dias_vision):
    # FASE 40: Sustituimos la llamada directa a Yahoo por nuestra función con Caché
    df_raw = descargar_datos_cacheados(ticker, "3y")
    
    if len(df_raw) < 200: return None, None, None
    
    df = calcular_indicadores(df_raw)
    df_train = df.copy()
    df_train['Target'] = np.where(df_train['Close'].shift(-dias_vision) > df_train['Close'] * 1.01, 1, 0)
    df_train = df_train.dropna()
    
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    if len(df_train) < 50: return None, None, None
        
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(df_train[pistas], df_train['Target'])
    
    hoy = df.iloc[-1]
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    
    pesos = model.feature_importances_ * 100
    importancias = dict(zip(pistas, pesos))
    
    return prob, hoy, importancias

def ejecutar_simulacion_parking(ticker, dias, dias_vision):
    # FASE 40: Sustituimos la llamada directa en el simulador también
    df_raw = descargar_datos_cacheados(ticker, "3y")
    df = calcular_indicadores(df_raw)
    
    liquidez = capital_total
    en_posicion = False
    acciones_compradas = 0
    precio_maximo = 0
    intereses_acumulados = 0.0
    curva_capital = []
    ops = 0
    umbral_final = umbral_base + st.session_state['auto_bias']
    rendimiento_diario = (apy_liquidez / 100) / 365
    
    for i in range(len(df) - dias, len(df)):
        hoy = df.iloc[i]
        
        if en_posicion:
            precio_maximo = max(precio_maximo, hoy['Close'])
            stop_pct = hoy['ATR_pct'] * multiplicador_atr
            precio_stop = precio_maximo * (1 - (stop_pct / 100))
            
            if hoy['Close'] <= precio_stop or i == len(df) - 1:
                liquidez += (acciones_compradas * hoy['Close']) - comision_fija
                en_posicion = False
                acciones_compradas = 0
                precio_maximo = 0
        else:
            interes_hoy = liquidez * rendimiento_diario
            liquidez += interes_hoy
            intereses_acumulados += interes_hoy
            
            if i < len(df) - dias_vision:
                estudio = df.iloc[:i]
                df_train = estudio.copy()
                df_train['Target'] = np.where(df_train['Close'].shift(-dias_vision) > df_train['Close'] * 1.01, 1, 0)
                df_train = df_train.dropna()
                if len(df_train) >= 50:
                    modelo = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                    modelo.fit(df_train[['Retorno', 'Volatilidad', 'RSI']], df_train['Target'])
                    prob = modelo.predict_proba(estudio[['Retorno', 'Volatilidad', 'RSI']].iloc[-1:]) [0][1] * 100
                    
                    pasa_macro = (hoy['Close'] > hoy['Media_200']) if filtro_macro else True
                    if prob >= umbral_final and hoy['Volatilidad'] > 0.5 and pasa_macro and hoy['RSI'] < 70:
                        ops += 1
                        en_posicion = True
                        precio_maximo = hoy['Close']
                        inv = liquidez * 0.2 
                        liquidez -= comision_fija
                        acciones_compradas = inv / hoy['Close']
                        liquidez -= inv
                        
        valor_cartera = liquidez + (acciones_compradas * hoy['Close']) if en_posicion else liquidez
        curva_capital.append(valor_cartera)
        
    return curva_capital, ops, intereses_acumulados

# ------------------------------------------------------------------------------
# 4. FUNCIONES DE TELEGRAM
# ------------------------------------------------------------------------------
def enviar_alerta(mensaje):
    url = f"https://api.telegram.org/bot{token_input}/sendMessage"
    payload = {"chat_id": chat_id_input, "text": mensaje, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

# ------------------------------------------------------------------------------
# 5. DASHBOARD PRINCIPAL
# ------------------------------------------------------------------------------
st.title("🏦 Portal IA: Ecosistema Local On-Premise")

tab1, tab2, tab3 = st.tabs(["🚀 Escáner de Liderazgo (HOY)", "🔬 Auditoría Algorítmica", "📊 Business Intelligence (BI)"])

umbral_f = umbral_base + st.session_state['auto_bias']

# ... (El código de tab1 y tab2 se mantiene exactamente igual) ...

# NUEVO: FASE 41 - La Pestaña de Business Intelligence (Simulación Tableau)
with tab3:
    st.subheader("📊 Cuadro de Mando Integral (Tableau AI Simulator)")
    st.write("Arrastra aquí tus archivos `trade_log_IA.csv`. El motor de BI transformará los datos crudos en inteligencia visual.")
    
    archivos_subidos = st.file_uploader("Sube tus archivos de auditoría CSV", type="csv", accept_multiple_files=True)
    
    if archivos_subidos:
        st.success(f"✅ {len(archivos_subidos)} archivo(s) ingerido(s) en el motor de BI.")
        
        lista_dfs = []
        for archivo in archivos_subidos:
            df_temp = pd.read_csv(archivo)
            lista_dfs.append(df_temp)
            
        df_bi = pd.concat(lista_dfs, ignore_index=True)
        df_bi = df_bi.drop_duplicates()
        
        # --- PANEL SUPERIOR: KPIs INSTITUCIONALES ---
        st.write("### 📈 KPIs de la Cartera (Key Performance Indicators)")
        total_analizados = len(df_bi)
        ordenes_compra = len(df_bi[df_bi['Order_Type'] == 'BUY'])
        capital_asignado_total = df_bi[df_bi['Order_Type'] == 'BUY']['Allocated_Capital_EUR'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Análisis Totales", total_analizados)
        c2.metric("Oportunidades Ejecutadas", ordenes_compra)
        c3.metric("Capital Teórico Asignado", f"{capital_asignado_total:.2f} €", help="Suma de capital propuesto en las órdenes BUY.")
        
        st.divider()
        
        # --- PANEL CENTRAL: VISUALIZACIONES AVANZADAS (PLOTLY) ---
        st.write("### 🧠 Inteligencia Visual")
        
        col_graf_1, col_graf_2 = st.columns(2)
        
        with col_graf_1:
            # Gráfico 1: Distribución de Decisiones (Pie Chart)
            # ¿Qué porcentaje del tiempo la IA nos manda a liquidez vs invertir?
            fig_pie = px.pie(
                df_bi, 
                names='Order_Type', 
                title='Distribución de Órdenes (Defensa vs Ataque)',
                color='Order_Type',
                color_discrete_map={'HOLD_CASH': '#8B0000', 'BUY': '#006400'},
                hole=0.4 # Lo hace formato "Donut"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_graf_2:
            # Gráfico 2: El Cerebro de la IA (Explicabilidad XAI)
            # ¿Cuáles son los motivos por los que la IA toma decisiones?
            conteo_motivos = df_bi['AI_Reasoning_XAI'].value_counts().reset_index()
            conteo_motivos.columns = ['Motivo (Indicador)', 'Frecuencia']
            
            fig_bar = px.bar(
                conteo_motivos, 
                x='Frecuencia', 
                y='Motivo (Indicador)', 
                orientation='h',
                title='Motores de Decisión Algorítmica',
                color='Frecuencia',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # --- PANEL INFERIOR: MAPA DE CALOR DE CONVICCIÓN ---
        st.write("### 🔥 Mapa de Convicción por Activo")
        # Visualizamos qué activos despiertan más seguridad en la IA, incluso si no compramos
        fig_scatter = px.scatter(
            df_bi, 
            x='Date', 
            y='AI_Probability', 
            color='Ticker',
            size='AI_Probability',
            hover_data=['Order_Type', 'AI_Reasoning_XAI'],
            title='Evolución de la Convicción Institucional en el Tiempo',
            labels={'AI_Probability': 'Probabilidad IA (%)', 'Date': 'Fecha del Análisis'}
        )
        # Añadimos una línea roja marcando nuestro umbral de seguridad
        fig_scatter.add_hline(y=umbral_f, line_dash="dot", line_color="red", annotation_text="Umbral de Compra")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("Ver Datos Crudos (Base de Datos Local)"):
            st.dataframe(df_bi)
