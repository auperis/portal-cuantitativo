# ==============================================================================
# ARQUITECTURA FASE 47: EL SIMULACRO DE INCENDIO (ROBOT ACELERADO)
# Objetivo: Forzar la ejecución, usar todos los núcleos y enviar Telegram.
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import schedule 
import os

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL ROBOT
# ------------------------------------------------------------------------------
TOKEN_TELEGRAM = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID = "1063578190"

# ¡SABOTAJE ACTIVADO! Umbral bajado al 1.0% para forzar que compre algo
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

# ------------------------------------------------------------------------------
# 2. MOTOR DE COMUNICACIONES
# ------------------------------------------------------------------------------
def enviar_alerta_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error enviando Telegram: {e}")

# ------------------------------------------------------------------------------
# 3. EL CEREBRO DE LA IA
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

def mision_escaneo_diario():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Radar Automático...")
    
    resultados_csv = []
    operaciones_hoy = 0
    
    for tick, nombre in UNIVERSO:
        print(f"  -> Analizando {tick}...")
        df_raw = yf.Ticker(tick).history(period="3y")
        
        if len(df_raw) < 200:
            continue
            
        df = calcular_indicadores(df_raw)
        df_train = df.copy()
        df_train['Target'] = np.where(df_train['Close'].shift(-DIAS_VISION) > df_train['Close'] * 1.01, 1, 0)
        df_train = df_train.dropna()
        
        pistas = ['Retorno', 'Volatilidad', 'RSI']
        if len(df_train) < 50:
            continue
            
        # ¡ACELERADOR V8 INSTALADO AQUÍ!
        model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42,)
        model.fit(df_train[pistas], df_train['Target'])
        
        hoy = df.iloc[-1]
        prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
        
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
                f"📦 *Orden:* Invertir `{inversion:.2f} €` ({acciones:.4f} uds).\n"
                f"🪂 *Paracaídas ATR:* `{precio_stop:.2f} $`"
            )
            enviar_alerta_telegram(msg)
            
        fecha_hoy = datetime.now().strftime("%Y-%m-%d")
        resultados_csv.append({
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
        
        time.sleep(1) 

    if resultados_csv:
        df_ordenes = pd.DataFrame(resultados_csv)
        nombre_archivo = f"trade_log_IA_{datetime.now().strftime('%Y%m%d')}.csv"
        df_ordenes.to_csv(nombre_archivo, index=False)
        print(f"\n[OK] Análisis completado. Archivo '{nombre_archivo}' guardado.")
        
        if operaciones_hoy == 0:
            enviar_alerta_telegram("🛡️ *PICAYO IA:* Mercado escaneado. 0 oportunidades detectadas. Manteniendo liquidez.")

# ------------------------------------------------------------------------------
# 5. EL RELOJ (CRON JOB)
# ------------------------------------------------------------------------------
print("🤖 Robot Ejecutor Iniciado. Ejecutando simulacro...")

schedule.every().day.at("22:15").do(mision_escaneo_diario)

if __name__ == "__main__":
    # CRISTAL ROTO: El robot se ejecutará nada más arrancar
    mision_escaneo_diario() 
    
    while True:
        schedule.run_pending()
        time.sleep(60)
