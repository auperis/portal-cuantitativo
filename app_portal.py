# ==============================================================================
# MÓDULO 6: GESTIÓN DE RIESGO CUANTITATIVA (EL ESCUDO)
# Objetivo: Calcular matemáticamente el tamaño de la posición para no arruinarnos.
# Restricción: Cuenta de 1.000 € (Solo arriesgamos el 2% por operación).
# ==============================================================================

def calcular_tamaño_posicion(capital_total, precio_accion, stop_loss_porcentaje, riesgo_maximo_porcentaje=2.0):
    """
    Esta función es nuestro 'Compartimento Estanco'.
    
    ¿Qué hace?
    1. Calcula cuánto dinero máximo podemos perder en esta operación (El 2% de 1.000€ = 20€).
    2. Mira nuestro "Stop-Loss" (el paracaídas: a qué % de caída venderemos automáticamente asumiendo el error).
    3. Calcula exactamente CUÁNTAS acciones debemos comprar para que, si salta el paracaídas, 
       solo hayamos perdido esos 20€.
    """
    print("🛡️ INICIANDO PROTOCOLO DE GESTIÓN DE RIESGO 🛡️")
    print(f"Capital Total de la Cartera: {capital_total} €")
    
    # 1. ¿Cuánto es lo máximo que nos permitimos perder si la IA se equivoca?
    riesgo_en_euros = capital_total * (riesgo_maximo_porcentaje / 100)
    print(f"Límite de pérdida (Compartimento estanco): {riesgo_en_euros} €")
    
    # 2. ¿Cuánto dinero perdemos POR ACCIÓN si cae hasta nuestro paracaídas (stop-loss)?
    # Ejemplo: Si la acción vale 100€ y nuestro stop-loss es del 5%, perderemos 5€ por acción.
    riesgo_por_accion = precio_accion * (stop_loss_porcentaje / 100)
    
    # 3. EL CÁLCULO MÁGICO (Position Sizing)
    # Dividimos nuestra pérdida máxima total entre la pérdida por acción.
    # Siguiendo el ejemplo: 20€ / 5€ = Podemos comprar 4 acciones.
    numero_acciones = riesgo_en_euros / riesgo_por_accion
    
    # Redondeamos hacia abajo, porque no podemos comprar medias acciones en muchos brokers
    numero_acciones_real = int(numero_acciones)
    
    # Calculamos cuánto dinero de nuestros 1.000€ vamos a invertir en total
    capital_a_invertir = numero_acciones_real * precio_accion
    
    print("\n--- INSTRUCCIONES DE EJECUCIÓN (BROKER) ---")
    print(f"Precio actual del activo: {precio_accion:.2f} €")
    print(f"Colocar Stop-Loss (Paracaídas automático) a un: -{stop_loss_porcentaje}% de caída")
    print(f"-> COMPRAR: {numero_acciones_real} acciones.")
    print(f"-> CAPITAL A INVERTIR: {capital_a_invertir:.2f} €")
    print(f"-> LIQUIDEZ RESTANTE: {(capital_total - capital_a_invertir):.2f} € (Guardado seguro)")
    print("-------------------------------------------")
    print(f"Si la IA acierta, ganamos. Si la IA falla y salta el stop-loss,")
    print(f"solo habremos perdido {riesgo_en_euros} €. El submarino sigue a flote.")
    
    return numero_acciones_real, capital_a_invertir

# ==============================================================================
# PRUEBA DEL SISTEMA
# ==============================================================================
# Imaginemos que la IA nos ha dado LUZ VERDE hoy para una acción que cuesta 150€.
# Decidimos que si la acción cae un 5%, asumiremos que la IA se equivocó y venderemos.
if __name__ == "__main__":
    acciones_a_comprar, inversion = calcular_tamaño_posicion(
        capital_total=1000,          # Nuestra restricción de capital
        precio_accion=150.0,         # Precio imaginario de la acción hoy
        stop_loss_porcentaje=5.0     # Nuestro nivel de dolor por operación
    )
