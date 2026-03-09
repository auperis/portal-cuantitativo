# ==============================================================================
# FASE 45: CÁPSULA BASE (DOCKERFILE UNIVERSAL)
# Objetivo: Crear el entorno perfecto y aislado (Python + Librerías)
# ==============================================================================

# 1. Sistema Operativo Base (Completo, sin '-slim')
FROM python:3.9

# 2. Establecer el directorio de trabajo dentro de la cápsula
WORKDIR /app

# 3. Copiar la lista de herramientas
COPY requirements.txt .

# 3.5 NUEVO PASO CLAVE: Actualizar la "Llave Inglesa" (pip) antes de instalar nada
RUN pip install --upgrade pip

# 4. Instalar las herramientas matemáticas y de IA
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO nuestro código (app_portal.py y robot_ejecutor.py) a la cápsula
COPY . .

# 6. Crear la carpeta interna del "Montacargas"
RUN mkdir -p /app/datos_bitacora
