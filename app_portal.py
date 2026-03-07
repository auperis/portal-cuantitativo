# ==============================================================================
# ARQUITECTURA FASE 33: EXPANSIÓN DEL UNIVERSO (SECTOR ROTATION)
# Objetivo: Aumentar la superficie de escaneo agrupando activos por sectores
# para encontrar oportunidades ocultas cuando los índices principales descansan.
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
# 1. CONFIGURACIÓN VISUAL
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Radar Expandido", layout="wide", page_icon="🌍")

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
# NUEVO: Selector de Universo Invertible
escanear_indices = st.sidebar.checkbox("Índices Globales (SPY, QQQ, IWM)", value=True)
escanear_sectores = st.sidebar.checkbox("Sectores (XLK, XLF, XLV, XLE)", value=True)
escanear_refugios = st.sidebar.checkbox("Refugios/Cripto (GLD, TLT, BTC-USD)", value=True)

st.sidebar.divider()
st.sidebar.header("🎛️ Calibración IA y Filtros")
dias_vision_ia = st.sidebar.slider("Visión IA (Días)", 1, 15, 5)
umbral_base = st.sidebar.slider("Umbral Probabilidad (%)", 50.0, 70.0, 52.0)
multiplicador_atr = st.sidebar.slider("Multiplicador ATR", 1.0, 5.0, 2.0)
filtro_macro = st.sidebar.checkbox("Lente Macro (Media 200)", value=True)

st.sidebar.divider()
comision_fija = st.sidebar.number_input("Comisión Broker (€)", value=1.0)
capital_total = st.sidebar.number_input("Capital Total (€)", value=1000)
max_exposicion = st.sidebar.slider("Exposición Máxima (%)", 5.0, 40.0, 25.0)

# ------------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# ------------------------------------------------------------------------------
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
    df_raw = yf.Ticker(ticker).history(period="3y")
    if len(df_raw) < 200: return None, None
    
    df = calcular_indicadores(df_raw)
    df_train = df.copy()
    df_train['Target'] = np.where(df_train['Close'].shift(-dias_vision) > df_train['Close'] * 1.01, 1, 0)
    df_train = df_train.dropna()
    
    pistas = ['Retorno', 'Volatilidad', 'RSI']
    if len(df_train) < 50: return None, None
        
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(df_train[pistas], df_train['Target'])
    
    hoy = df.iloc[-1]
    prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
    return prob, hoy

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
st.title("🌍 Portal IA: Radar de Rotación Sectorial")

tab1, tab2 = st.tabs(["🚀 Radar en Vivo (HOY)", "🔬 Memoria del Sistema"])

umbral_f = umbral_base + st.session_state['auto_bias']

with tab1:
    st.markdown(f"### Escáner Institucional | Umbral Exigido: **{umbral_f:.1f}%**")
    
    if st.button("🚀 INICIAR ESCÁNER GLOBAL", type="primary"):
        # Construimos el universo según lo seleccionado en la barra lateral
        universo = []
        if escanear_indices: universo.extend([("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("IWM", "Russell 2000")])
        if escanear_sectores: universo.extend([("XLK", "Tecnología"), ("XLF", "Financiero"), ("XLV", "Salud"), ("XLE", "Energía")])
        if escanear_refugios: universo.extend([("GLD", "Oro"), ("TLT", "Bonos 20A"), ("BTC-USD", "Bitcoin")])
        
        if not universo:
            st.error("⚠️ Debes seleccionar al menos un grupo de activos en la barra lateral.")
        else:
            resultados = []
            operaciones_encontradas = 0
            barra_progreso = st.progress(0)
            
            for i, (tick, nombre) in enumerate(universo):
                with st.spinner(f"Analizando {nombre} ({tick})..."):
                    prob, datos_hoy = entrenar_ia_radar(tick, dias_vision_ia)
                    
                    if prob is not None:
                        precio = datos_hoy['Close']
                        vol_ok = datos_hoy['Volatilidad'] > 0.5
                        macro_ok = (precio > datos_hoy['Media_200']) if filtro_macro else True
                        rsi_ok = datos_hoy['RSI'] < 70
                        
                        estado = "🔴 DESCARTADO"
                        accion = "Esperar / Liquidez"
                        
                        if prob >= umbral_f and vol_ok and macro_ok and rsi_ok:
                            estado = "🟢 SEÑAL CONFIRMADA"
                            operaciones_encontradas += 1
                            
                            rango = 100 - umbral_f
                            factor = min(max((prob - umbral_f) / rango, 0), 1) if rango > 0 else 0
                            exp_pct = 0.05 + (factor * (max_exposicion/100 - 0.05))
                            inversion = capital_total * exp_pct
                            acciones = inversion / precio
                            
                            stop_distancia = datos_hoy['ATR_pct'] * multiplicador_atr
                            precio_stop = precio * (1 - (stop_distancia/100))
                            
                            accion = f"Comprar {acciones:.2f} uds"
                            
                            msg = (
                                f"🌍 *SEÑAL DE ROTACIÓN SECTORIAL*\n\n"
                                f"Sector/Activo: `{nombre} ({tick})`\n"
                                f"Convicción IA: `{prob:.1f}%` 📈\n"
                                f"Precio Actual: `{precio:.2f} $`\n\n"
                                f"📦 *Orden:* Invertir `{inversion:.2f} €`.\n"
                                f"🪂 *Stop ATR:* `{precio_stop:.2f} $`"
                            )
                            if activar_alertas: enviar_alerta(msg)
                                
                        elif prob >= umbral_f:
                            estado = "🟡 EN OBSERVACIÓN"
                            if not macro_ok: accion = "Tendencia Macro Bajista"
                            elif not rsi_ok: accion = "Sobrecomprado (RSI>70)"
                            elif not vol_ok: accion = "Volatilidad Insuficiente"
                            
                        resultados.append({
                            "Categoría": nombre,
                            "Ticker": tick,
                            "Prob IA": f"{prob:.1f}%",
                            "Estado": estado,
                            "RSI": f"{datos_hoy['RSI']:.1f}",
                            "Instrucción": accion
                        })
                
                barra_progreso.progress((i + 1) / len(universo))
                
            st.subheader(f"📊 Resultados del Escáner ({len(universo)} Activos)")
            df_res = pd.DataFrame(resultados)
            
            # Usamos Streamlit native styling para resaltar colores
            def color_estado(val):
                if "CONFIRMADA" in val: return 'background-color: #004d00; color: white'
                elif "OBSERVACIÓN" in val: return 'background-color: #664d00; color: white'
                elif "DESCARTADO" in val: return 'background-color: #4d0000; color: white'
                return ''
                
            st.dataframe(df_res.style.applymap(color_estado, subset=['Estado']), use_container_width=True)
            
            if operaciones_encontradas > 0:
                st.success(f"🎯 Se encontraron {operaciones_encontradas} oportunidades tras expandir el universo.")
            else:
                st.info("🛡️ Disciplina Institucional: 10 mercados analizados. 0 oportunidades claras. Protegiendo los 1.000 € hoy.")

with tab2:
    st.write("Módulo de calibración en reposo. Parámetros actuales optimizados.")
