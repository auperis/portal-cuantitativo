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

# NUEVO: Añadida la tercera pestaña "Diario Local"
tab1, tab2, tab3 = st.tabs(["🚀 Escáner de Liderazgo (HOY)", "🔬 Auditoría Algorítmica", "📂 Diario Local (Ledger)"])

umbral_f = umbral_base + st.session_state['auto_bias']

with tab1:
    st.markdown(f"### Muro de Inteligencia | Umbral Exigido: **{umbral_f:.1f}%**")
    
    if st.button("🔎 EJECUTAR ANÁLISIS DE FUERZA RELATIVA"):
        universo = []
        if escanear_indices: universo.extend([("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("IWM", "Russell 2000")])
        if escanear_sectores: universo.extend([("XLK", "Tecnología"), ("XLF", "Financiero"), ("XLV", "Salud"), ("XLE", "Energía")])
        if escanear_refugios: universo.extend([("GLD", "Oro"), ("TLT", "Bonos 20A"), ("BTC-USD", "Bitcoin")])
        
        if not universo:
            st.error("⚠️ Debes seleccionar al menos un grupo de activos.")
        else:
            resultados = []
            log_ordenes = []
            operaciones_encontradas = 0
            barra_progreso = st.progress(0)
            
            for i, (tick, nombre) in enumerate(universo):
                with st.spinner(f"Midiendo fuerza relativa de {nombre} ({tick})..."):
                    prob, datos_hoy, pesos_ia = entrenar_ia_radar(tick, dias_vision_ia)
                    
                    if prob is not None:
                        precio = datos_hoy['Close']
                        vol_ok = datos_hoy['Volatilidad'] > 0.5
                        macro_ok = (precio > datos_hoy['Media_200']) if filtro_macro else True
                        rsi_ok = datos_hoy['RSI'] < 70
                        
                        estado = "🔴 DESCARTADO"
                        accion = "Liquidez"
                        acciones = 0.0
                        precio_stop = 0.0
                        inversion = 0.0
                        
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
                            
                        motivo_principal = max(pesos_ia, key=pesos_ia.get)
                        peso_motivo = pesos_ia[motivo_principal]
                            
                        resultados.append({
                            "Activo": nombre,
                            "Ticker": tick,
                            "Valor_Prob": prob, 
                            "Convicción IA": f"{prob:.1f}%",
                            "Motor de Decisión (XAI)": f"{motivo_principal} ({peso_motivo:.1f}%)",
                            "Estado": estado,
                            "RSI": f"{datos_hoy['RSI']:.1f}",
                            "Instrucción": accion
                        })
                        
                        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
                        tipo_orden = "BUY" if "CONFIRMADA" in estado else "HOLD_CASH"
                        
                        log_ordenes.append({
                            "Date": fecha_hoy,
                            "Ticker": tick,
                            "Order_Type": tipo_orden,
                            "Quantity_Shares": round(acciones, 4),
                            "Allocated_Capital_EUR": round(inversion, 2),
                            "Limit_Price_USD": round(precio, 2),
                            "Trailing_Stop_USD": round(precio_stop, 2),
                            "AI_Probability": round(prob, 2),
                            "AI_Reasoning_XAI": motivo_principal
                        })
                
                barra_progreso.progress((i + 1) / len(universo))
            
            df_res = pd.DataFrame(resultados)
            df_res = df_res.sort_values(by="Valor_Prob", ascending=False).reset_index(drop=True)
            df_res_mostrar = df_res.drop(columns=['Valor_Prob'])
            
            lider = df_res.iloc[0]
            st.info(f"🏆 **El Activo más fuerte hoy es {lider['Activo']}** con un {lider['Convicción IA']} de convicción. Su estado actual es: {lider['Estado']}.")
            
            st.subheader(f"📊 Ranking de Inteligencia ({len(universo)} Activos)")
            
            def color_estado(val):
                if "CONFIRMADA" in val: return 'background-color: #004d00; color: white'
                elif "OBSERVACIÓN" in val: return 'background-color: #664d00; color: white'
                elif "DESCARTADO" in val: return 'background-color: #4d0000; color: white'
                return ''
                
            st.dataframe(df_res_mostrar.style.applymap(color_estado, subset=['Estado']), use_container_width=True)
            
            st.divider()
            st.subheader("🏦 Exportación On-Premise (InvestGlass Ready)")
            
            df_ordenes = pd.DataFrame(log_ordenes)
            csv_data = df_ordenes.to_csv(index=False).encode('utf-8')
            
            if operaciones_encontradas > 0:
                st.success(f"🎯 Se han generado {operaciones_encontradas} órdenes. Descarga el archivo para guardarlo en tu registro local.")
            else:
                st.info("🛡️ El archivo CSV generado hoy contiene puras órdenes de HOLD_CASH. Es vital guardarlo para tu estadística.")
                
            st.download_button(
                label="📥 Descargar Archivo de Órdenes Diario (CSV)",
                data=csv_data,
                file_name=f"trade_log_IA_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

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
            c_a.metric("Operaciones Realizadas", n_ops)
            c_b.metric("Intereses por Esperar", f"+{int_acumulados:.2f} €", delta="Dinero 100% Pasivo")
            c_c.metric("Beneficio Neto Total", f"{beneficio_total:.2f} €", delta=f"Capital Final: {curva[-1]:.2f} €")

# NUEVO: FASE 38 - La Pestaña de Diario Local
with tab3:
    st.subheader("📂 Lector de Bitácora Local (Data Prep)")
    st.write("Arrastra aquí los archivos CSV que has descargado en días anteriores. El sistema los leerá localmente sin enviarlos a internet.")
    
    # Cargador de archivos que acepta múltiples CSV
    archivos_subidos = st.file_uploader("Sube tus archivos trade_log_IA.csv", type="csv", accept_multiple_files=True)
    
    if archivos_subidos:
        st.success(f"✅ {len(archivos_subidos)} archivo(s) cargado(s) en la memoria local.")
        
        # Combinamos todos los archivos en un solo DataFrame gigante
        lista_dfs = []
        for archivo in archivos_subidos:
            df_temp = pd.read_csv(archivo)
            lista_dfs.append(df_temp)
            
        df_historico = pd.concat(lista_dfs, ignore_index=True)
        
        # Limpiamos duplicados (por si el usuario sube el archivo de hoy dos veces por error)
        df_historico = df_historico.drop_duplicates()
        
        st.write("### 📊 Resumen Estadístico de tu Cartera")
        
        # Cálculos de Análisis
        total_analizados = len(df_historico)
        ordenes_compra = len(df_historico[df_historico['Order_Type'] == 'BUY'])
        ordenes_espera = len(df_historico[df_historico['Order_Type'] == 'HOLD_CASH'])
        
        # El "Ratio de Acción" nos dice qué porcentaje del tiempo realmente disparamos el francotirador
        ratio_accion = (ordenes_compra / total_analizados) * 100 if total_analizados > 0 else 0
        
        col_1, col_2, col_3 = st.columns(3)
        col_1.metric("Análisis Registrados", total_analizados, help="Total de activos escaneados en todos tus archivos.")
        col_2.metric("Órdenes de Compra Emitidas", ordenes_compra)
        col_3.metric("Ratio de Acción", f"{ratio_accion:.1f}%", help="Mide qué tan selectivo es tu sistema.")
        
        st.divider()
        st.write("### 🧠 ¿Qué piensa tu IA? (Patrones de Decisión)")
        
        # Mostramos los motivos principales por los que la IA descarta el mercado
        st.write("Frecuencia del Motor de Decisión (XAI):")
        conteo_motivos = df_historico['AI_Reasoning_XAI'].value_counts()
        st.bar_chart(conteo_motivos)
        
        # Mostramos la tabla unificada para que el usuario pueda filtrar
        st.write("Registro Completo (Filtrable):")
        st.dataframe(df_historico)
        
        st.info("💡 Este paso es el pre-procesamiento de datos (Data Prep) necesario antes de conectar tus resultados a una herramienta de Business Intelligence como **Tableau AI**.")
