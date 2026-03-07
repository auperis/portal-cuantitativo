# ==============================================================================
# ARQUITECTURA FASE 34: ESCÁNER DE FUERZA RELATIVA (RELATIVE STRENGTH)
# Objetivo: Ordenar el universo de activos por la convicción de la IA para
# mantener la vigilancia estratégica incluso en días de "Cero Operaciones".
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
# 1. CONFIGURACIÓN VISUAL (MODO INSTITUCIONAL)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Portal IA - Fuerza Relativa", layout="wide", page_icon="🏆")

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

# NUEVO: Fase 35 - El Parking Inteligente
st.sidebar.divider()
st.sidebar.header("💰 Gestión de Liquidez (Parking)")
apy_liquidez = st.sidebar.slider("Fondo Monetario (APY %)", 0.0, 10.0, 3.5, help="Rendimiento anual por tener el dinero esperando sin operar.")
dias_simulacion = st.sidebar.selectbox("Días de Simulación Auditoría", [90, 180, 365], index=1)

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

def ejecutar_simulacion_parking(ticker, dias, dias_vision):
    df = calcular_indicadores(yf.Ticker(ticker).history(period="3y"))
    
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
            # LA MAGIA DEL PARKING: Ganamos dinero solo por esperar
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
st.title("🏆 Portal IA: Ranking de Fuerza Relativa")

tab1, tab2 = st.tabs(["🚀 Escáner de Liderazgo (HOY)", "🔬 Auditoría"])

umbral_f = umbral_base + st.session_state['auto_bias']

with tab1:
    st.markdown(f"### Muro de Inteligencia | Umbral Exigido: **{umbral_f:.1f}%**")
    
    # CORRECCIÓN DE INTERFAZ: Se elimina type="primary" para quitar el color rojo de alerta.
    # Ahora es un botón neutral institucional.
    if st.button("🔎 EJECUTAR ANÁLISIS DE FUERZA RELATIVA"):
        universo = []
        if escanear_indices: universo.extend([("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("IWM", "Russell 2000")])
        if escanear_sectores: universo.extend([("XLK", "Tecnología"), ("XLF", "Financiero"), ("XLV", "Salud"), ("XLE", "Energía")])
        if escanear_refugios: universo.extend([("GLD", "Oro"), ("TLT", "Bonos 20A"), ("BTC-USD", "Bitcoin")])
        
        if not universo:
            st.error("⚠️ Debes seleccionar al menos un grupo de activos.")
        else:
            resultados = []
            operaciones_encontradas = 0
            barra_progreso = st.progress(0)
            
            for i, (tick, nombre) in enumerate(universo):
                with st.spinner(f"Midiendo fuerza relativa de {nombre} ({tick})..."):
                    prob, datos_hoy = entrenar_ia_radar(tick, dias_vision_ia)
                    
                    if prob is not None:
                        precio = datos_hoy['Close']
                        vol_ok = datos_hoy['Volatilidad'] > 0.5
                        macro_ok = (precio > datos_hoy['Media_200']) if filtro_macro else True
                        rsi_ok = datos_hoy['RSI'] < 70
                        
                        estado = "🔴 DESCARTADO"
                        accion = "Liquidez"
                        
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
                            
                        elif prob >= umbral_f:
                            estado = "🟡 EN OBSERVACIÓN"
                            if not macro_ok: accion = "Tendencia Bajista (Media 200)"
                            elif not rsi_ok: accion = "RSI Sobrecomprado"
                            elif not vol_ok: accion = "Sin Volatilidad"
                            
                        resultados.append({
                            "Activo": nombre,
                            "Ticker": tick,
                            # Guardamos la probabilidad como número para poder ordenar la tabla matemáticamente
                            "Valor_Prob": prob, 
                            "Convicción IA": f"{prob:.1f}%",
                            "Estado": estado,
                            "RSI": f"{datos_hoy['RSI']:.1f}",
                            "Instrucción": accion
                        })
                
                barra_progreso.progress((i + 1) / len(universo))
            
            # MAGIA INSTITUCIONAL: Ordenamos los resultados de mayor a menor convicción
            df_res = pd.DataFrame(resultados)
            df_res = df_res.sort_values(by="Valor_Prob", ascending=False).reset_index(drop=True)
            
            # Eliminamos la columna de uso interno antes de mostrarla
            df_res_mostrar = df_res.drop(columns=['Valor_Prob'])
            
            # Mostramos el líder indiscutible
            lider = df_res.iloc[0]
            st.info(f"🏆 **El Activo más fuerte hoy es {lider['Activo']}** con un {lider['Convicción IA']} de convicción. Su estado actual es: {lider['Estado']}.")
            
            st.subheader(f"📊 Ranking de Inteligencia ({len(universo)} Activos)")
            
            def color_estado(val):
                if "CONFIRMADA" in val: return 'background-color: #004d00; color: white'
                elif "OBSERVACIÓN" in val: return 'background-color: #664d00; color: white'
                elif "DESCARTADO" in val: return 'background-color: #4d0000; color: white'
                return ''
                
            st.dataframe(df_res_mostrar.style.applymap(color_estado, subset=['Estado']), use_container_width=True)
            
            if operaciones_encontradas == 0:
                st.write("---")
                st.warning("🛡️ **Análisis del Arquitecto:** Aunque no hay señales confirmadas de compra, el ranking superior te indica qué activos están acumulando fuerza. Vigila a los que están 'EN OBSERVACIÓN', ya que podrían dar señal verde mañana.")

with tab2:
    st.subheader("🔬 Auditoría de Memoria y Parking Inteligente")
    st.write("Mientras el mercado está 'Descartado', tu liquidez genera intereses pasivos.")
    
    activo_sim = st.selectbox("Selecciona un activo para auditar (Ej. QQQ, SPY):", ["QQQ", "SPY", "GLD"])
    
    if st.button(f"🏁 Simular {dias_simulacion} Días con APY del {apy_liquidez}%"):
        with st.spinner("Calculando operaciones e intereses pasivos acumulados..."):
            curva, n_ops, int_acumulados = ejecutar_simulacion_parking(activo_sim, dias_simulacion, dias_vision_ia)
            
            st.line_chart(curva)
            beneficio_total = curva[-1] - capital_total
            
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Operaciones Realizadas", n_ops, help="Veces que salimos del parking para cazar una tendencia.")
            c_b.metric("Intereses por Esperar", f"+{int_acumulados:.2f} €", delta="Dinero 100% Pasivo")
            c_c.metric("Beneficio Neto Total", f"{beneficio_total:.2f} €", delta=f"Capital Final: {curva[-1]:.2f} €")
            
            if int_acumulados > 0:
                st.success(f"✅ Has ganado **{int_acumulados:.2f} €** simplemente por tener paciencia y no forzar operaciones cuando la IA decía 'DESCARTADO'.")
