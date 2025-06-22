import numpy as np
import igraph as ig
import time
import os
import logging
import pickle
from math import radians, cos, sin, asin, sqrt

# --- Constantes y Configuración ---
NUM_USERS_REFERENCE = 10_000_000
LOCATION_TXT_FILE = 'D:\\ADA\\dataset\\10_million_location.txt'
USER_TXT_FILE = 'D:\\ADA\\dataset\\10_million_user.txt'
OUTPUT_DIR = './processed_data_igraph'
GRAPH_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network.igraph.pkl")
LOCATIONS_NPY_FILE = os.path.join(OUTPUT_DIR, "social_network_locations.npy")
ID_MAP_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_id_mappings.pkl")
COMMUNITIES_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_communities.pkl")
LOG_FILE = os.path.join(OUTPUT_DIR, "graph_processing.log")

# --- Configuración del Logging ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# --- Funciones Auxiliares ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia en kilómetros entre dos puntos (lat, lon) en la Tierra."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

class GraphProcessor:
    """
    Clase para encapsular la carga, construcción, análisis y guardado
    de un grafo de red social a gran escala usando igraph.
    """
    def __init__(self, user_filepath, location_filepath):
        self.user_filepath = user_filepath
        self.location_filepath = location_filepath
        self.graph = None
        self.id2idx = None
        self.idx2id = None
        self.locations = None
        self.communities = None

    def _load_locations(self):
        """Carga las ubicaciones de los usuarios en un array de NumPy."""
        logging.info(f"Iniciando carga de ubicaciones desde {self.location_filepath}")
        start_time = time.time()
        try:
            # Asumimos que el archivo de locaciones tiene N líneas, correspondiendo a usuarios 1..N
            self.locations = np.loadtxt(self.location_filepath, delimiter=',', dtype=np.float32)
            if self.locations.shape[1] != 2:
                raise ValueError("El archivo de ubicaciones no tiene 2 columnas.")
            elapsed = time.time() - start_time
            logging.info(f"Se cargaron {len(self.locations)} ubicaciones en {elapsed:.2f} segundos.")
        except FileNotFoundError:
            logging.error(f"Error Crítico: Archivo de ubicaciones no encontrado en {self.location_filepath}.")
            self.locations = None
        except Exception as e:
            logging.error(f"Error Crítico al cargar ubicaciones: {e}", exc_info=True)
            self.locations = None

    def _build_graph_two_pass(self):
        """
        Construye el grafo igraph usando un enfoque optimizado de dos pasadas.
        CORREGIDO: El ID de usuario de origen es el número de línea.
        """
        if not os.path.exists(self.user_filepath):
            logging.error(f"Error Crítico: Archivo de usuarios no encontrado en {self.user_filepath}.")
            return

        logging.info("Iniciando pasada 1: Recolectando IDs de usuario únicos...")
        start_time = time.time()
        all_ids = set()
        try:
            with open(self.user_filepath, 'r') as f:
                # Usamos enumerate para obtener el número de línea (empezando en 0)
                for i, line in enumerate(f):
                    # CAMBIO CLAVE: El ID del usuario de origen es el número de línea + 1
                    source_id_orig = i + 1
                    all_ids.add(source_id_orig)

                    # Los números en la línea son los IDs de los usuarios de destino
                    target_ids_str = [p.strip() for p in line.split(',') if p.strip()]
                    if not target_ids_str:
                        continue
                    
                    # Añadimos los IDs de destino al conjunto de todos los IDs
                    all_ids.update(map(int, target_ids_str))

        except (ValueError, IndexError) as e:
            logging.error(f"Error parseando IDs en la primera pasada: {e}. Revisa el formato del archivo.", exc_info=True)
            return

        sorted_ids = sorted(list(all_ids))
        self.id2idx = {uid: i for i, uid in enumerate(sorted_ids)}
        self.idx2id = sorted_ids

        num_nodes = len(self.idx2id)
        logging.info(f"Pasada 1 completada en {time.time() - start_time:.2f}s. Encontrados {num_nodes} nodos únicos.")

        def edge_generator():
            with open(self.user_filepath, 'r') as f:
                # Usamos enumerate de nuevo para mantener la consistencia
                for i, line in enumerate(f):
                    # CAMBIO CLAVE: El ID de origen es el número de línea + 1
                    source_id_orig = i + 1
                    
                    # Obtenemos el índice del grafo para el nodo de origen
                    # No es necesario comprobar si existe, porque lo añadimos en la pasada 1
                    source_idx = self.id2idx[source_id_orig]
                    
                    # Parseamos los IDs de destino
                    target_ids_str = [p.strip() for p in line.split(',') if p.strip()]
                    if not target_ids_str:
                        continue

                    try:
                        # CAMBIO CLAVE: Iteramos sobre TODOS los números de la línea como destinos
                        for target_str in target_ids_str:
                            target_id_orig = int(target_str)
                            # Creamos una arista si el destino también es un nodo válido
                            if target_id_orig in self.id2idx:
                                yield (source_idx, self.id2idx[target_id_orig])
                    except (ValueError, KeyError):
                        # Ignora líneas o partes mal formateadas
                        continue

        logging.info("Iniciando pasada 2: Construyendo grafo desde el generador de aristas...")
        start_time = time.time()
        self.graph = ig.Graph(n=num_nodes, edges=edge_generator(), directed=True)
        logging.info(f"Grafo construido en {time.time() - start_time:.2f}s.")
        logging.info(f"Grafo final: {self.graph.vcount()} nodos, {self.graph.ecount()} aristas.")
        self.graph.simplify(multiple=True, loops=True)
        logging.info(f"Grafo simplificado (eliminando duplicados y bucles): {self.graph.vcount()} nodos, {self.graph.ecount()} aristas.")

    def run_build_pipeline(self):
        """Ejecuta el pipeline completo de carga y construcción."""
        self._load_locations()
        self._build_graph_two_pass()
        if self.graph:
            self._add_attributes_to_graph()

    def _add_attributes_to_graph(self):
        """Añade atributos de ID original y ubicación a los vértices del grafo."""
        if self.locations is None:
            logging.warning("No hay datos de ubicación para añadir al grafo. Se usarán coordenadas (0,0).")
        
        logging.info("Asignando atributos a los nodos del grafo...")
        self.graph.vs["original_id"] = self.idx2id
        
        latitudes = np.zeros(self.graph.vcount(), dtype=np.float32)
        longitudes = np.zeros(self.graph.vcount(), dtype=np.float32)
        
        # Mapeamos las ubicaciones a los nodos del grafo
        # El archivo de ubicaciones corresponde a los usuarios 1, 2, 3...
        # Su índice en el array `self.locations` es `id - 1`
        for i, original_id in enumerate(self.idx2id):
            # Comprobamos si el ID tiene una ubicación válida
            if 1 <= original_id <= len(self.locations):
                loc_idx = original_id - 1
                latitudes[i] = self.locations[loc_idx, 0]
                longitudes[i] = self.locations[loc_idx, 1]
        
        self.graph.vs["latitude"] = latitudes.tolist()
        self.graph.vs["longitude"] = longitudes.tolist()
        logging.info("Atributos de ID original y ubicación asignados.")


    def analyze_basic_metrics(self):
        """Realiza un análisis básico del grafo."""
        if not self.graph: logging.warning("Grafo no disponible para análisis básico."); return
        logging.info("Realizando análisis básico del grafo...")
        start = time.time()
        num_nodes, num_edges = self.graph.vcount(), self.graph.ecount()
        logging.info(f"Análisis - Nodos: {num_nodes}, Aristas: {num_edges}")
        if num_nodes == 0: logging.warning("Grafo vacío, análisis detenido."); return
        logging.info(f"Análisis - Densidad del grafo: {self.graph.density():.4e}")
        components = self.graph.components(mode='weak')
        logging.info(f"Análisis - Número de componentes conectados (débil): {len(components)}")
        if len(components) > 0:
            giant = components.giant()
            logging.info(f"Análisis - Tamaño del componente conectado más grande: {giant.vcount()} nodos ({giant.vcount()/num_nodes*100:.2f}%)")
        logging.info(f"Tiempo de análisis básico: {time.time() - start:.2f} segundos.")

    def detect_communities(self, method='louvain'):
        """Detecta comunidades usando un algoritmo especificado."""
        if not self.graph: logging.error("El grafo no está construido."); return
        logging.info(f"Iniciando detección de comunidades con el método '{method}'...")
        start_time = time.time()
        g_undirected = self.graph.as_undirected(mode='collapse')
        if method == 'louvain':
            self.communities = g_undirected.community_multilevel()
        else:
            logging.error(f"Método de detección de comunidades '{method}' no reconocido. Use 'louvain'.")
            return
        self.graph.vs["community"] = self.communities.membership
        elapsed = time.time() - start_time
        logging.info(f"Detección de comunidades completada en {elapsed:.2f}s. "
                     f"Comunidades: {len(self.communities)}, Modularidad: {self.communities.modularity:.4f}")

    def analyze_shortest_path(self, sample_size=1000):
        """
        Estima la longitud promedio del camino más corto de forma SECUENCIAL y con uso de memoria optimizado.
        Este método es más lento que el paralelo pero consume significativamente menos memoria.
        """
        if not self.graph:
            logging.error("El grafo no está construido."); return

        logging.info(f"Estimando (en SECUENCIAL y con memoria optimizada) la longitud promedio del camino más corto con una muestra de {sample_size} nodos...")
        start_time = time.time()
        components = self.graph.components(mode='weak')
        giant = components.giant()
        if giant.vcount() < 2:
            logging.warning("El componente gigante es demasiado pequeño para el análisis."); return
        if giant.vcount() < sample_size:
            sample_size = giant.vcount()
            logging.warning(f"Tamaño de la muestra reducido a {sample_size} (tamaño del componente gigante).")

        sampled_vertices_indices = np.random.choice(giant.vcount(), size=sample_size, replace=False)
        
        ## CAMBIO: En lugar de una lista, usamos un acumulador para la suma y otro para el conteo.
        total_path_lengths_sum = 0.0
        total_paths_count = 0

        logging.info("Iniciando bucle de cálculo secuencial...")

        for i, v_idx in enumerate(sampled_vertices_indices):
            # Esta línea sigue consumiendo memoria temporalmente, pero se libera en cada iteración.
            paths_from_v = giant.distances(source=v_idx)[0]
            
            ## CAMBIO: Iteramos sobre los caminos y actualizamos los acumuladores en lugar de guardar los caminos en una lista.
            # Esto evita el MemoryError.
            for p in paths_from_v:
                if p != float('inf') and p > 0:
                    total_path_lengths_sum += p
                    total_paths_count += 1
            
            if (i + 1) % 100 == 0:
                elapsed_min = (time.time() - start_time) / 60
                logging.info(f"  ... procesados {i + 1}/{sample_size} nodos de muestra. (Tiempo transcurrido: {elapsed_min:.2f} min)")

        ## CAMBIO: Calculamos el promedio usando la suma y el conteo.
        average_path_length = total_path_lengths_sum / total_paths_count if total_paths_count > 0 else float('inf')
        
        elapsed = time.time() - start_time
        logging.info(f"Estimación secuencial completada en {elapsed:.2f}s.")
        logging.info(f"  - Longitud promedio estimada del camino más corto: {average_path_length:.4f}")
        return average_path_length
    
    def calculate_mst_on_distance(self):
        """Calcula el Árbol de Expansión Mínima (MST) basado en la distancia geográfica."""
        if not self.graph or "latitude" not in self.graph.vs.attributes():
            logging.error("Grafo o atributos de ubicación no disponibles para calcular el MST."); return
        logging.info("Iniciando cálculo del Árbol de Expansión Mínima (MST) basado en distancia.")
        start_time = time.time()
        
        # El MST solo tiene sentido en un grafo no dirigido
        g_undirected = self.graph.as_undirected(mode='collapse')
        
        giant_comp = g_undirected.components(mode='weak').giant()
        if giant_comp.vcount() < 2:
            logging.warning("Componente gigante demasiado pequeño para calcular MST."); return
        
        logging.info(f"Calculando MST en el componente gigante ({giant_comp.vcount()} nodos).")
        weights = [
            haversine_distance(
                giant_comp.vs[e.source]["latitude"], giant_comp.vs[e.source]["longitude"],
                giant_comp.vs[e.target]["latitude"], giant_comp.vs[e.target]["longitude"]
            ) for e in giant_comp.es
        ]
        
        giant_comp.es["weight"] = weights
        logging.info("Calculando MST con algoritmo optimizado de igraph...")
        mst = giant_comp.spanning_tree(weights=giant_comp.es["weight"], return_tree=True)
        total_mst_length = sum(mst.es["weight"])
        elapsed = time.time() - start_time
        logging.info(f"Cálculo del MST completado en {elapsed:.2f}s.")
        logging.info(f"  - Longitud total del MST (suma de distancias en km): {total_mst_length:,.2f} km")
        return mst, total_mst_length

    def save_data(self):
        """Guarda todos los artefactos procesados en archivos."""
        if self.graph:
            logging.info(f"Guardando grafo en {GRAPH_PKL_FILE}...")
            with open(GRAPH_PKL_FILE, 'wb') as f: pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.id2idx and self.idx2id:
            logging.info(f"Guardando mapeos de ID en {ID_MAP_PKL_FILE}...")
            with open(ID_MAP_PKL_FILE, 'wb') as f: pickle.dump({'id2idx': self.id2idx, 'idx2id': self.idx2id}, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.locations is not None:
             logging.info(f"Guardando array de ubicaciones en {LOCATIONS_NPY_FILE}...")
             np.save(LOCATIONS_NPY_FILE, self.locations)
        if self.communities:
            logging.info(f"Guardando resultados de comunidades en {COMMUNITIES_PKL_FILE}...")
            # Guardamos la membresía como una lista simple para mayor compatibilidad
            membership_list = self.communities.membership
            with open(COMMUNITIES_PKL_FILE, 'wb') as f: pickle.dump(membership_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Todos los datos procesados han sido guardados.")

    def get_followed_users(self, user_id):
        """Devuelve una lista de los IDs de los usuarios seguidos por el usuario dado."""
        if not self.graph or not self.id2idx:
            logging.error("El grafo o los mapeos de ID no están disponibles."); return []
        if user_id not in self.id2idx:
            logging.warning(f"El usuario con ID {user_id} no se encuentra en el grafo."); return []
        vertex_idx = self.id2idx[user_id]
        followed_indices = self.graph.successors(vertex_idx)
        return [self.idx2id[idx] for idx in followed_indices]

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    if not all(os.path.exists(f) for f in [LOCATION_TXT_FILE, USER_TXT_FILE]):
        logging.critical(f"Error: No se encontraron los archivos de entrada. Revisa las rutas:\n  {LOCATION_TXT_FILE}\n  {USER_TXT_FILE}")
    else:
        processor = GraphProcessor(USER_TXT_FILE, LOCATION_TXT_FILE)
        processor.run_build_pipeline()

        if processor.graph:
            processor.analyze_basic_metrics()
            processor.detect_communities(method='louvain')
            
            # Llamada a la función SECUENCIAL para el análisis de caminos
            processor.analyze_shortest_path(sample_size=1000)
            
            processor.calculate_mst_on_distance()
            processor.save_data()

            #Demostración de la función de consulta
            logging.info("--- EJEMPLO DE USO DE get_followed_users ---")
            test_user_id = processor.idx2id[710]
            logging.info(f"Buscando a quién sigue el usuario con ID original: {test_user_id}...")
            followed_users_list = processor.get_followed_users(test_user_id)
            if followed_users_list:
                logging.info(f"El usuario {test_user_id} sigue a {len(followed_users_list)} usuarios. Primeros seguidos: {followed_users_list[:15]}")
            else:
                logging.info(f"El usuario {test_user_id} no sigue a nadie.")
        

            logging.info("--- ANÁLISIS COMPLETADO EXITOSAMENTE ---")
        else:
            logging.critical("La construcción del grafo falló. El script terminará.")