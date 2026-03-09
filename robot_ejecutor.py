# ==============================================================================
# ARQUITECTURA PICAYO IA - ROBOT EJECUTOR (VERSIÓN ESTABLE CON CHIVATOS)
# Objetivo: Escaneo diario, alertas Telegram y diagnóstico en tiempo real.
# ==============================================================================

import time
import schedule
import os
import requests
from datetime import datetime

# --- CHIVATOS DE ARRANQUE (TINTE FLUORESCENTE) ---
print(">>> [PUNTO 1] EL ROBOT HA DESPERTADO. INICIANDO SISTEMA...", flush=True)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

print(">>> [PUNTO 2] LIBRERÍAS MATEMÁTICAS Y ANTENA YAHOO CARGADAS.", flush=True)

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL ROBOT (DATOS DEL ARQUITECTO)
# ------------------------------------------------------------------------------
TOKEN_TELEGRAM = "8713410900:AAF-6ZxBDBwRcDDdVYV1CPEIxM7adJL4tVA"
CHAT_ID = "1063578190"

# UMBRAL DE SABOTAJE: 1.0% para forzar que el robot envíe señales hoy mismo.
UMBRAL_COMPRA = 1.0 

DIAS_VISION = 3
MULTIPLICADOR_ATR = 1.5
CAPITAL_TOTAL = 1000

# Universo de 12 activos (Liderazgo, Cripto y Protección Inversa)
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
        print(f"Error enviando Telegram: {e}", flush=True)

# ------------------------------------------------------------------------------
# 3. EL CEREBRO DE LA IA Y CÁLCULOS
# ------------------------------------------------------------------------------
def calcular_indicadores(df):
    d = df.copy()
    d['Retorno'] = d['Close'].pct_change() * 100
    d['Media_200'] = d['Close'].rolling(200).mean() 
    d['Volatilidad'] = d['Retorno'].rolling(10).std()
    
    # RSI (Fuerza Relativa)
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    d['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # ATR (Turbulencia)
    d['High_Low'] = d['High'] - d['Low']
    d['High_PrevClose'] = np.abs(d['High'] - d['Close'].shift(1))
    d['Low_PrevClose'] = np.abs(d['Low'] - d['Close'].shift(1))
    d['True_Range'] = d[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    d['ATR_pct'] = (d['True_Range'].rolling(14).mean() / d['Close']) * 100
    return d.dropna()

def mision_escaneo_diario():
    print(f"\n>>> [PUNTO 3] [{datetime.now().strftime('%H:%M:%S')}] INICIANDO RADAR DIARIO...", flush=True)
    
    resultados_csv = []
    operaciones_hoy = 0
    
    for tick, nombre in UNIVERSO:
        print(f"  -> Analizando {tick} ({nombre})...", flush=True)
        try:
            df_raw = yf.Ticker(tick).history(period="3y")
            if len(df_raw) < 200: continue
            
            df = calcular_indicadores(df_raw)
            df_train = df.copy()
            df_train['Target'] = np.where(df_train['Close'].shift(-DIAS_VISION) > df_train['Close'] * 1.01, 1, 0)
            df_train = df_train.dropna()
            
            pistas = ['Retorno', 'Volatilidad', 'RSI']
            
            # ENTRENAMIENTO IA (Sin n_jobs para evitar colapso en Docker Desktop)
            model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
            model.fit(df_train[pistas], df_train['Target'])
            
            hoy = df.iloc[-1]
            prob = model.predict_proba(df[pistas].iloc[-1:]) [0][1] * 100
            
            precio = hoy['Close']
            # Filtros de seguridad básicos
            vol_ok = hoy['Volatilidad'] > 0.1
            rsi_ok = hoy['RSI'] < 80
            
            tipo_orden = "HOLD_CASH"
            inversion = 0.0
            
            if prob >= UMBRAL_COMPRA and vol_ok and rsi_ok:
                operaciones_hoy += 1
                tipo_orden = "BUY"
                
                # Gestión de Capital (Entre 5% y 25% del total)
                rango = 100 - UMBRAL_COMPRA
                factor = min(max((prob - UMBRAL_COMPRA) / rango, 0), 1) if rango > 0 else 0
                exp_pct = 0.05 + (factor * (0.25 - 0.05))
                inversion = CAPITAL_TOTAL * exp_pct
                
                # Stop Loss ATR
                precio_stop = precio * (1 - (hoy['ATR_pct'] * MULTIPLICADOR_ATR / 100))
                
                msg = (
                    f"🤖 *PICAYO IA: SEÑAL CONFIRMADA*\n\n"
                    f"Activo: `{nombre} ({tick})`\n"
                    f"Probabilidad: `{prob:.1f}%` (Umbral: {UMBRAL_COMPRA}%)\n"
                    f"Precio Actual: `{precio:.2f} $`\n\n"
                    f"📦 *Orden:* Invertir `{inversion:.2f} €`\n"
                    f"🪂 *Stop-Loss (ATR):* `{precio_stop:.2f} $`"
                )
                enviar_alerta_telegram(msg)
            
            resultados_csv.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Ticker": tick,
                "Order_Type": tipo_orden,
                "AI_Probability": round(prob, 2),
                "Capital_Allocated": round(inversion, 2)
            })
            
        except Exception as e:
            print(f"  [!] Error en {tick}: {e}", flush=True)

    # Guardar bitácora local
    if resultados_csv:
        df_log = pd.DataFrame(resultados_csv)
        nombre_archivo = f"log_robot_{datetime.now().strftime('%Y%m%d')}.csv"
        df_log.to_csv(nombre_archivo, index=False)
        print(f">>> [PUNTO 4] ESCANEO FINALIZADO. LOG GUARDADO: {nombre_archivo}", flush=True)

# ------------------------------------------------------------------------------
# 4. CICLO DE VIDA DEL ROBOT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # EJECUCIÓN INMEDIATA (Simulacro de Incendio)
    mision_escaneo_diario()
    
    # PROGRAMACIÓN DIARIA (La hora de la verdad)
    schedule.every().day.at("22:15").do(mision_escaneo_diario)
    
    print(">>> [PUNTO 5] ROBOT EN MODO ESPERA (DORMIDO HASTA LAS 22:15)", flush=True)
    while True:
        schedule.run_pending()
        time.sleep(60)
