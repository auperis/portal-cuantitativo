# ==============================================================================
# ARQUITECTURA FASE 47: EL SIMULACRO (CON TINTE FLUORESCENTE)
# ==============================================================================

# --- LOS CHIVATOS (TINTE FLUORESCENTE) ---
print(">>> [PUNTO 1] EL ROBOT HA DESPERTADO. INICIANDO SISTEMA...", flush=True)

import time
print(">>> [PUNTO 2] LIBRERÍAS DE TIEMPO CARGADAS.", flush=True)

import schedule
import os
import requests
from datetime import datetime
print(">>> [PUNTO 3] LIBRERÍAS DE COMUNICACIÓN CARGADAS.", flush=True)

print(">>> [PUNTO 4] CARGANDO MATEMÁTICAS PESADAS (Scikit-Learn/Pandas)...", flush=True)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
print(">>> [PUNTO 5] MATEMÁTICAS LISTAS.", flush=True)

print(">>> [PUNTO 6] CONECTANDO ANTENA DE YAHOO FINANCE...", flush=True)
import yfinance as yf
print(">>> [PUNTO 7] ANTENA CONECTADA. TODAS LAS LIBRERÍAS OK.", flush=True)

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL ROBOT
# ------------------------------------------------------------------------------
TOKEN_TELEGRAM = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID = "1063578190"

# ¡SABOTAJE ACTIVADO! Umbral bajado al 1.0%
UMBRAL_COMPRA = 1.0 
DIAS_VISION = 3
MULTIPLICADOR_ATR = 1.5
CAPITAL_TOTAL = 1000

UNIVERSO = [
    ("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("IWM", "Russell 2000"),
    ("XLK", "Tecnología"), ("XLF", "Financiero"), ("XLV", "Salud"), ("XLE", "Energía"),
    ("GLD", "Oro"), ("TLT", "Bonos 20A"), ("BTC-USD", "Bitcoin"),
    ("SH", "Inverso S&P 500"), ("SQQQ", "Inverso Nasdaq")
]

def enviar_alerta_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error enviando Telegram: {e}", flush=True)

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

def mision_escaneo_diario():
    print(f"\n>>> [PUNTO 8] [{datetime.now().strftime('%H:%M:%S')}] INICIANDO RADAR DE ACTIVOS...", flush=True)
    
    resultados_csv = []
    operaciones_hoy = 0
    
    for tick, nombre in UNIVERSO:
        print(f"  -> Descargando datos de {tick}...", flush=True)
        try:
            df_raw = yf.Ticker(tick).history(period="3y")
        except Exception as e:
            print(f"  [!] Error descargando {tick}: {e}", flush=True)
            continue
            
        if len(df_raw) < 200:
            print(f"  -> {tick} descartado (pocos datos).", flush=True)
            continue
            
        print(f"  -> Entrenando IA para {tick}...", flush=True)
        df = calcular_indicadores(df_raw)
        df_train = df.copy()
        df_train['Target'] = np.where(df_train['Close'].shift(-DIAS_VISION) > df_train['Close'] * 1.01, 1, 0)
        df_train = df_train.dropna()
        
        pistas = ['Retorno', 'Volatilidad', 'RSI']
        if len(df_train) < 50:
            continue
            
        # SIN n_jobs PARA EVITAR COLAPSOS
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
        model.fit(df_train[pistas], df_train['Target'])
        
        hoy = df.iloc[-1]
        prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
        print(f"  -> {tick} procesado. Probabilidad: {prob:.1f}%", flush=True)
        
        pesos = model.feature_importances_ * 100
        importancias = dict(zip(pistas, pesos))
        motivo_principal = max(importancias, key=importancias.get)
        
        precio = hoy['Close']
        vol_ok = hoy['Volatilidad'] > 0.5
        rsi_ok = hoy['RSI'] < 70
        
        acciones = 0.0
        precio_stop = 0.0
        inversion = 0.0
        tipo_orden = "HOLD_CASH"
        
        if prob >= UMBRAL_COMPRA and vol_ok and rsi_ok:
            operaciones_hoy += 1
            tipo_orden = "BUY"
            
            rango = 100 - UMBRAL_COMPRA
            factor = min(max((prob - UMBRAL_COMPRA) / rango, 0), 1) if rango > 0 else 0
            exp_pct = 0.05 + (factor * (0.25 - 0.05))
            inversion = CAPITAL_TOTAL * exp_pct
            acciones = inversion / precio
            
            stop_distancia = hoy['ATR_pct'] * MULTIPLICADOR_ATR
            precio_stop = precio * (1 - (stop_distancia/100))
            
            msg = (
                f"🤖 *PICAYO IA: SEÑAL CONFIRMADA*\n\n"
                f"Activo: `{nombre} ({tick})`\n"
                f"Probabilidad: `{prob:.1f}%`\n"
                f"Precio: `{precio:.2f} $`\n\n"
                f"📦 *Orden:* Invertir `{inversion:.2f} €`"
            )
            print(f"  -> ¡ENVIANDO TELEGRAM PARA {tick}!", flush=True)
            enviar_alerta_telegram(msg)
            
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        resultados_csv.append({
            "Date": fecha_hoy,
            "Ticker": tick,
            "Order_Type": tipo_orden,
            "AI_Probability": round(prob, 2),
        })
        time.sleep(1) 

    if resultados_csv:
        print("\n>>> [PUNTO 9] GUARDANDO BITÁCORA CSV...", flush=True)
        df_ordenes = pd.DataFrame(resultados_csv)
        df_ordenes.to_csv(f"trade_log_IA_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        
        if operaciones_hoy == 0:
            enviar_alerta_telegram("🛡️ *PICAYO IA:* 0 oportunidades detectadas.")
    print(">>> [PUNTO 10] MISIÓN CUMPLIDA. ROBOT A LA ESPERA.", flush=True)

# ------------------------------------------------------------------------------
# 5. EL RELOJ (CRON JOB)
# ------------------------------------------------------------------------------
schedule.every().day.at("22:15").do(mision_escaneo_diario)

if __name__ == "__main__":
    mision_escaneo_diario() # Ejecución inmediata (Cristal roto)
    while True:
        schedule.run_pending()
        time.sleep(60)
