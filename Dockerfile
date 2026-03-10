# 1. Sistema Operativo Base
FROM python:3.9

# 2. Establecer el directorio
WORKDIR /app

# 3. Copiar herramientas
COPY requirements.txt .

# 3.5 Actualizar pip
RUN pip install --upgrade pip

# 4. Instalar librerías
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO el código
COPY . .

# 6. Crear carpeta de logs
RUN mkdir -p /app/datos_bitacora
