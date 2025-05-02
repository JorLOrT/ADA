# ADA
Repositorio para el curso de análisis y diseño de algoritmos.

## Integrantes

- Jhordan Steven Octavio Huamani Huamani
- Jorge Luis Ortiz Castañeda

# Procesador de Redes Sociales a Gran Escala

Este proyecto implementa un sistema optimizado para cargar, procesar y analizar datos de redes sociales a gran escala (hasta 10 millones de usuarios). Utiliza técnicas eficientes de manejo de memoria y procesamiento por lotes para construir un grafo dirigido que representa la red social, junto con datos de geolocalización de los usuarios.

## 🚀 Características Principales

- **Procesamiento Eficiente:** Carga y procesa 10 millones de registros de usuarios y ubicaciones geográficas con optimización de memoria.
- **Construcción de Grafo:** Utiliza [igraph](https://igraph.org/python/) para generar un grafo dirigido de la red social optimizado para análisis.
- **Análisis Exploratorio:** Incluye análisis básico del grafo con métricas como grados de entrada/salida, componentes conectados y densidad.
- **Gestión Robusta de Errores:** Sistema exhaustivo de manejo de excepciones y validación de datos.
- **Logging Detallado:** Registro completo de todas las operaciones y errores para facilitar depuración.

## 📋 Requisitos

- Python 3.6+
- NumPy
- python-igraph
- Espacio en disco suficiente para archivos de datos (estimado ~5GB)
- Memoria RAM recomendada: 16GB+ (optimizado para usar menos memoria que soluciones naive)

## 🗂️ Estructura de Datos

El programa trabaja con dos archivos de entrada principales:

1. **Archivo de Ubicaciones (`10_million_location.txt`)**:
   - Cada línea contiene coordenadas geográficas (latitud,longitud)
   - El número de línea corresponde al ID de usuario
   - Formato: `lat,lon` (ej: `40.7128,-74.0060`)

2. **Archivo de Usuarios (`10_million_user.txt`)**:
   - Cada línea contiene relaciones de seguimiento
   - Formato: `user_id,follower1_id,follower2_id,...`
   - El primer número es el ID del usuario, seguido por los IDs de los usuarios que sigue

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/usuario/procesador-redes-sociales.git
cd procesador-redes-sociales

# Instalar dependencias
pip install numpy python-igraph

# Crear directorio para datos procesados
mkdir -p ./processed_data
```

## 🔧 Configuración

Ajusta las variables en la sección de configuración del script:

```python
# --- Configuration ---
NUM_USERS = 10_000_000
LOCATION_TXT_FILE = './dataset/10_million_location.txt'
USER_TXT_FILE = './dataset/10_million_user.txt'
OUTPUT_DIR = './processed_data'
```

## 🚀 Uso

Ejecuta el script principal:

```bash
python graph_builder.py
```

El proceso completo incluye:
1. Carga de datos de geolocalización
2. Construcción del grafo social optimizada para memoria
3. Análisis exploratorio básico
4. Guardado de datos procesados para uso futuro

## 📊 Salidas

El programa genera los siguientes archivos:

1. **`social_network_graph_10M.igraph.pkl`**: Grafo completo serializado
2. **`social_network_locations_10M.pkl`**: Diccionario de ubicaciones por ID
3. **`social_network_idx2id_10M.pkl`**: Mapeo de índices internos a IDs originales
4. **`social_network_id2idx_10M.pkl`**: Mapeo de IDs originales a índices internos
5. **`graph_builder.log`**: Archivo de registro detallado

## 📝 Detalles Técnicos

### Optimizaciones de Memoria
- **Procesamiento en dos pasadas**: Primero recolecta IDs únicos, luego genera aristas
- **Uso de generadores**: Evita listas de aristas en memoria durante la construcción
- **Mapeo de IDs**: Convierte IDs originales (potencialmente grandes) a índices compactos
- **NumPy para ubicaciones**: Utiliza arrays tipados para eficiencia

### Análisis Exploratorio de Datos
El script realiza automáticamente un análisis básico que incluye:
- Densidad del grafo
- Distribución de grados de entrada/salida
- Usuarios con más seguidores y seguidos
- Usuarios aislados (sin seguidores o seguidos)
- Componentes conectados
- Estadísticas de la red social

## 🔍 Manejo de Errores y Validación
- Validación de coordenadas geográficas
- Manejo de datos malformados o faltantes
- Tratamiento de IDs inválidos o inconsistentes
- Control de memoria para evitar desbordamientos
- Logging detallado para diagnóstico y monitoreo

## 📖 Ejemplo de Análisis

El análisis básico produce información como:

```
Análisis - Nodos: 8,743,291, Aristas: 42,351,602
Análisis - Densidad del grafo: 5.5432e-07
Análisis - Usuario con más seguidores: ID 573829 (12,503 seguidores)
Análisis - Usuario que sigue a más personas: ID 2938471 (4,291 seguidos)
Análisis - Componentes conectados (débil): 423
Análisis - Tamaño del componente conectado más grande: 8,701,523 nodos (99.52%)
```

## 🔄 Extensiones Futuras

- Análisis de comunidades
- Cálculo de medidas de centralidad (betweenness, closeness)
- Visualización de la red y geolocalización
- Paralelización de la carga y procesamiento

## 📜 Licencia

MIT



