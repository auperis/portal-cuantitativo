# ==============================================================================
# FASE 45: CÁPSULA BASE (DOCKERFILE UNIVERSAL)
# Objetivo: Crear el entorno perfecto y aislado (Python + Librerías)
# ==============================================================================

# 1. Sistema Operativo Base (Ligero y seguro)
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo dentro de la cápsula
WORKDIR /app

# 3. Copiar la lista de herramientas
COPY requirements.txt .

# 4. Instalar las herramientas matemáticas y de IA
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO nuestro código (app_portal.py y robot_ejecutor.py) a la cápsula
COPY . .

# 6. Crear la carpeta interna del "Montacargas"
RUN mkdir -p /app/datos_bitacora

# (Nota: El comando CMD de arranque ahora lo decide docker-compose.yml)
