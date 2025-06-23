import numpy as np
import igraph as ig
import time
import os
import logging
import pickle
from typing import List, Dict, Optional, Tuple

# --- Constantes y Configuración ---
NUM_USERS_REFERENCE = 10_000_000
LOCATION_TXT_FILE = 'D:\\ADA\\dataset\\10_million_location.txt'
USER_TXT_FILE = 'D:\\ADA\\dataset\\10_million_user.txt'
OUTPUT_DIR = './processed_data_igraph'
GRAPH_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network.igraph.pkl")
MST_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_mst.igraph.pkl")
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
def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

class GraphProcessor:
    def __init__(self, user_filepath: str, location_filepath: str):
        self.user_filepath = user_filepath
        self.location_filepath = location_filepath
        self.graph: Optional[ig.Graph] = None
        self.id2idx: Optional[Dict[int, int]] = None
        self.idx2id: Optional[List[int]] = None
        self.locations: Optional[np.ndarray] = None
        self.communities: Optional[ig.VertexClustering] = None

    def _load_locations(self) -> None:
        logging.info(f"Iniciando carga de ubicaciones desde {self.location_filepath}")
        start_time = time.time()
        try:
            self.locations = np.loadtxt(self.location_filepath, delimiter=',', dtype=np.float32)
            if self.locations.ndim != 2 or self.locations.shape[1] != 2:
                raise ValueError("El archivo de ubicaciones debe tener 2 columnas (lat, lon).")
            elapsed = time.time() - start_time
            logging.info(f"Se cargaron {len(self.locations)} ubicaciones en {elapsed:.2f} segundos.")
        except FileNotFoundError:
            logging.error(f"Error Crítico: Archivo de ubicaciones no encontrado en {self.location_filepath}.")
            self.locations = None
        except Exception as e:
            logging.error(f"Error Crítico al cargar ubicaciones: {e}", exc_info=True)
            self.locations = None

    def _build_graph_two_pass(self) -> None:
        if not os.path.exists(self.user_filepath):
            logging.error(f"Error Crítico: Archivo de usuarios no encontrado en {self.user_filepath}.")
            return
        logging.info("Iniciando pasada 1: Recolectando IDs de usuario únicos...")
        start_time = time.time()
        all_ids = set()
        try:
            with open(self.user_filepath, 'r') as f:
                for i, line in enumerate(f):
                    source_id_orig = i + 1
                    all_ids.add(source_id_orig)
                    target_ids_str = line.strip().split(',')
                    if target_ids_str and target_ids_str[0]:
                        all_ids.update(map(int, target_ids_str))
        except (ValueError, IndexError) as e:
            logging.error(f"Error parseando IDs en la primera pasada: {e}.", exc_info=True)
            return

        sorted_ids = sorted(list(all_ids))
        self.id2idx = {uid: i for i, uid in enumerate(sorted_ids)}
        self.idx2id = sorted_ids
        num_nodes = len(self.idx2id)
        logging.info(f"Pasada 1 completada en {time.time() - start_time:.2f}s. {num_nodes} nodos únicos encontrados.")

        def edge_generator():
            with open(self.user_filepath, 'r') as f:
                for i, line in enumerate(f):
                    source_id_orig = i + 1
                    source_idx = self.id2idx[source_id_orig]
                    target_ids_str = line.strip().split(',')
                    if not target_ids_str or not target_ids_str[0]: continue
                    for target_str in target_ids_str:
                        try:
                            target_id_orig = int(target_str)
                            if target_id_orig in self.id2idx:
                                yield (source_idx, self.id2idx[target_id_orig])
                        except (ValueError, KeyError):
                            continue

        logging.info("Iniciando pasada 2: Construyendo grafo desde el generador de aristas...")
        start_time = time.time()
        self.graph = ig.Graph(n=num_nodes, edges=edge_generator(), directed=True)
        logging.info(f"Grafo construido en {time.time() - start_time:.2f}s. ({self.graph.vcount()} nodos, {self.graph.ecount()} aristas).")
        
        logging.info("Simplificando grafo...")
        self.graph.simplify(multiple=True, loops=True)
        logging.info(f"Grafo simplificado: {self.graph.vcount()} nodos, {self.graph.ecount()} aristas.")

    def _add_attributes_to_graph_vectorized(self) -> None:
        if self.locations is None:
            logging.warning("No hay datos de ubicación. Se usarán coordenadas (0,0).")
            self.graph.vs["original_id"] = self.idx2id
            self.graph.vs["latitude"] = [0.0] * self.graph.vcount()
            self.graph.vs["longitude"] = [0.0] * self.graph.vcount()
            return
        logging.info("Asignando atributos a nodos del grafo...")
        start_time = time.time()
        self.graph.vs["original_id"] = self.idx2id
        original_ids_in_graph = np.array(self.idx2id, dtype=np.int32)
        latitudes = np.zeros(self.graph.vcount(), dtype=np.float32)
        longitudes = np.zeros(self.graph.vcount(), dtype=np.float32)
        max_loc_id = self.locations.shape[0]
        valid_mask = (original_ids_in_graph >= 1) & (original_ids_in_graph <= max_loc_id)
        valid_original_ids = original_ids_in_graph[valid_mask]
        valid_locations = self.locations[valid_original_ids - 1]
        latitudes[valid_mask] = valid_locations[:, 0]
        longitudes[valid_mask] = valid_locations[:, 1]
        self.graph.vs["latitude"] = latitudes.tolist()
        self.graph.vs["longitude"] = longitudes.tolist()
        logging.info(f"Atributos asignados en {time.time() - start_time:.2f}s.")

    def run_build_pipeline(self) -> None:
        self._load_locations()
        self._build_graph_two_pass()
        if self.graph:
            self._add_attributes_to_graph_vectorized()

    def analyze_basic_metrics(self) -> None:
        if not self.graph: logging.warning("Grafo no disponible para análisis."); return
        logging.info("Realizando análisis básico del grafo...")
        start = time.time()
        num_nodes, num_edges = self.graph.vcount(), self.graph.ecount()
        if num_nodes == 0: logging.warning("Grafo vacío."); return
        
        logging.info(f"Análisis - Nodos: {num_nodes:,}, Aristas: {num_edges:,}")
        logging.info(f"Análisis - Densidad: {self.graph.density():.4e}")
        
        wcc = self.graph.components(mode='weak')
        logging.info(f"Análisis - Componentes débilmente conectados (WCC): {len(wcc):,}")
        if wcc:
            giant_wcc = wcc.giant()
            logging.info(f"Análisis - WCC más grande: {giant_wcc.vcount():,} nodos ({giant_wcc.vcount()/num_nodes:.2%})")

        scc = self.graph.components(mode='strong')
        logging.info(f"Análisis - Componentes fuertemente conectados (SCC): {len(scc):,}")
        if scc:
            giant_scc = scc.giant()
            logging.info(f"Análisis - SCC más grande: {giant_scc.vcount():,} nodos ({giant_scc.vcount()/num_nodes:.2%})")

        logging.info(f"Tiempo de análisis básico: {time.time() - start:.2f}s.")

    def analyze_degree_centrality(self, top_n: int = 10) -> None:
        if not self.graph or not self.idx2id:
            logging.warning("Grafo o mapeo de IDs no disponible."); return
        
        logging.info(f"Analizando los {top_n} nodos con mayor grado (in/out)...")
        start_time = time.time()
        if self.graph.vcount() == 0: return

        in_degrees = np.array(self.graph.degree(mode='in'))
        top_in_indices = np.argsort(in_degrees)[-top_n:]
        logging.info("--- Nodos con mayor Grado de Entrada (Más 'Populares') ---")
        for idx in reversed(top_in_indices):
            logging.info(f"  - Usuario ID: {self.idx2id[idx]:<10} | Grado de Entrada: {in_degrees[idx]:,}")

        out_degrees = np.array(self.graph.degree(mode='out'))
        top_out_indices = np.argsort(out_degrees)[-top_n:]
        logging.info("--- Nodos con mayor Grado de Salida (Más 'Activos') ---")
        for idx in reversed(top_out_indices):
            logging.info(f"  - Usuario ID: {self.idx2id[idx]:<10} | Grado de Salida: {out_degrees[idx]:,}")
            
        logging.info(f"Análisis de grado completado en {time.time() - start_time:.2f}s.")

    def detect_communities(self, method: str = 'louvain') -> None:
        if not self.graph: logging.error("El grafo no está construido."); return
        logging.info(f"Iniciando detección de comunidades con '{method}'...")
        start_time = time.time()
        g_undirected = self.graph.as_undirected(mode='collapse')
        self.communities = g_undirected.community_multilevel()
        self.graph.vs["community"] = self.communities.membership
        elapsed = time.time() - start_time
        logging.info(f"Comunidades detectadas en {elapsed:.2f}s. Comunidades: {len(self.communities)}, Modularidad: {self.communities.modularity:.4f}")

    def analyze_shortest_path_native(self, sample_size: int = 2000) -> Optional[float]:
        if not self.graph: logging.error("El grafo no está construido."); return None
        logging.info(f"Estimando longitud de camino promedio con muestra de {sample_size}...")
        start_time = time.time()
        
        giant = self.graph.components(mode='weak').giant()
        if giant.vcount() < 2: logging.warning("Componente gigante muy pequeño."); return None
        if giant.vcount() < sample_size:
            sample_size = giant.vcount()
            logging.warning(f"Tamaño de la muestra reducido a {sample_size}.")

        sampled_vertices_indices = np.random.choice(giant.vcount(), size=sample_size, replace=False)
        total_path_lengths_sum, total_paths_count = 0.0, 0

        logging.info("Iniciando bucle de cálculo de distancias ...")
        for i, v_idx in enumerate(sampled_vertices_indices):
            paths_from_v = giant.distances(source=v_idx)[0]
            for p in paths_from_v:
                if p != float('inf') and p > 0:
                    total_path_lengths_sum += p
                    total_paths_count += 1
            if (i + 1) % 500 == 0:
                logging.info(f"  ... procesados {i + 1}/{sample_size} nodos de muestra.")

        if total_paths_count == 0: return float('inf')
        average_path_length = total_path_lengths_sum / total_paths_count
        elapsed = time.time() - start_time
        logging.info(f"Estimación de camino promedio completada en {elapsed:.2f}s.")
        logging.info(f"  - Longitud promedio estimada: {average_path_length:.4f}")
        return average_path_length

    def calculate_mst_on_distance_vectorized(self) -> Optional[Tuple[ig.Graph, float]]:
        if not self.graph or "latitude" not in self.graph.vs.attributes():
            logging.error("Grafo o atributos de ubicación no disponibles."); return None
        
        logging.info("Calculando MST...")
        start_time = time.time()
        
        g_undirected = self.graph.as_undirected(mode='collapse')
        giant = g_undirected.components(mode='weak').giant()
        if giant.vcount() < 2: logging.warning("Componente gigante muy pequeño para calcular MST."); return None
        
        edges = np.array(giant.get_edgelist(), dtype=np.int32)
        
        lats = giant.vs["latitude"]
        lons = giant.vs["longitude"]
        coords = np.column_stack((lats, lons)).astype(np.float32)

        lat1, lon1 = coords[edges[:, 0], 0], coords[edges[:, 0], 1]
        lat2, lon2 = coords[edges[:, 1], 0], coords[edges[:, 1], 1]
        weights = haversine_vectorized(lat1, lon1, lat2, lon2)
        giant.es["weight"] = weights
        mst = giant.spanning_tree(weights="weight", return_tree=True)
        total_mst_length = sum(mst.es["weight"])

        logging.info(f"Guardando grafo del MST en {MST_PKL_FILE}...")
        mst.write_pickle(MST_PKL_FILE)
        
        elapsed = time.time() - start_time
        logging.info(f"Cálculo de MST completado en {elapsed:.2f}s.")
        logging.info(f"  - Longitud total del MST: {total_mst_length:,.2f} km")
        return mst, total_mst_length

    def save_data(self) -> None:
        logging.info("Iniciando guardado de datos procesados...")
        if self.graph:
            logging.info(f"Guardando grafo principal en {GRAPH_PKL_FILE}...")
            self.graph.write_pickle(GRAPH_PKL_FILE)
        if self.id2idx and self.idx2id:
            logging.info(f"Guardando mapeos de ID en {ID_MAP_PKL_FILE}...")
            with open(ID_MAP_PKL_FILE, 'wb') as f: pickle.dump({'id2idx': self.id2idx, 'idx2id': self.idx2id}, f)
        if self.locations is not None:
             logging.info(f"Guardando array de ubicaciones en {LOCATIONS_NPY_FILE}...")
             np.save(LOCATIONS_NPY_FILE, self.locations)
        if self.communities:
            logging.info(f"Guardando resultados de comunidades en {COMMUNITIES_PKL_FILE}...")
            with open(COMMUNITIES_PKL_FILE, 'wb') as f: pickle.dump(self.communities, f)
        logging.info("Todos los datos procesados han sido guardados.")

if __name__ == "__main__":
    if not all(os.path.exists(f) for f in [LOCATION_TXT_FILE, USER_TXT_FILE]):
        logging.critical(f"Error: No se encontraron los archivos de entrada.")
    else:
        processor = GraphProcessor(USER_TXT_FILE, LOCATION_TXT_FILE)
        main_start_time = time.time()
        
        processor.run_build_pipeline()

        if processor.graph:
            processor.analyze_basic_metrics()
            processor.analyze_degree_centrality(top_n=10)
            processor.detect_communities(method='louvain')
            processor.analyze_shortest_path_native(sample_size=2000)
            processor.calculate_mst_on_distance_vectorized()
            processor.save_data()
            
            total_time = time.time() - main_start_time
            logging.info(f"--- ANÁLISIS COMPLETADO en {total_time/60:.2f} minutos ---")
        else:
            logging.critical("La construcción del grafo falló. El script terminará.")