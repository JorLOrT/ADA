import numpy as np
import igraph as ig
import time
import os
import logging
import pickle
from math import radians, sin, cos, sqrt, atan2

# =========================================================
# CONFIGURACIÓN Y CONSTANTES
# =========================================================
NUM_USERS = 10_000_000
LOCATION_TXT_FILE = './dataset/10_million_location.txt'
USER_TXT_FILE = './dataset/10_million_user.txt'
OUTPUT_DIR = './processed_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

GRAPH_IGRAPH_FILE = os.path.join(OUTPUT_DIR, "social_network_graph.igraph.pkl")
LOCATIONS_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_locations.pkl")
IDX2ID_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_idx2id.pkl")
ID2IDX_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_id2idx.pkl")
LOG_FILE = os.path.join(OUTPUT_DIR,"graph_builder.log")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [Builder] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def haversine(coord1, coord2):
    """
    Calcula la distancia Haversine entre dos puntos (lat, lon).
    Devuelve la distancia en kilómetros.
    """
    R = 6371.0 
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    if (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0):
        return 40000.0

    rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1

    a = sin(dlat / 2)**2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return max(distance, 0.1)

# =========================================================
def load_locations(filepath, num_users):
    """
    Carga las ubicaciones desde un archivo de texto a un array de NumPy.
    Cada línea debe tener el formato: lat,lon
    Devuelve un array de shape (num_users, 2)
    """
    logging.info(f"Iniciando carga de ubicaciones desde {filepath}")
    locations = np.zeros((num_users, 2), dtype=np.float32)
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_users:
                    break
                try:
                    lat, lon = map(float, line.strip().split(','))
                    locations[i] = [lat, lon]
                except (ValueError, IndexError):
                    locations[i] = [0.0, 0.0]
        logging.info("Carga de ubicaciones completada.")
        return locations
    except FileNotFoundError:
        logging.error(f"Error Cr\u00edtico: Archivo de ubicaciones no encontrado en {filepath}.")
        return None

# =========================================================
def build_weighted_graph(user_filepath, locations_np):
    """
    Construye el grafo social ponderado a partir de los archivos de datos.
    Devuelve: grafo igraph, id2idx, idx2id
    """
    if locations_np is None:
        return None, None, None

    logging.info("Primera pasada: Recolectando todos los IDs \u00fanicos...")
    user_ids = set()
    try:
        with open(user_filepath, 'r') as f:
            for line_num, line_content in enumerate(f, start=1):
                if line_num > NUM_USERS:
                    break
                user_ids.add(line_num) 
                
                dst_ids_str = line_content.strip().split(',')
                for id_str in dst_ids_str:
                    cleaned_id = id_str.strip()
                    if cleaned_id: 
                        user_ids.add(int(cleaned_id))
    except FileNotFoundError:
        logging.error(f"Error Cr\u00edtico: Archivo de usuarios no encontrado en {user_filepath}.")
        return None, None, None
    except ValueError as e:
        logging.error(f"Error de valor en la primera pasada, l\u00ednea {line_num}: {e}. Revisa el formato del archivo de usuarios.")
        return None, None, None

    sorted_user_ids = sorted(list(user_ids))
    id2idx = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
    idx2id = {idx: uid for uid, idx in id2idx.items()}
    num_unique_nodes = len(id2idx)
    logging.info(f"Mapeados {num_unique_nodes:,} IDs \u00fanicos a \u00edndices de grafo (0 a {num_unique_nodes-1}).")

    logging.info("Segunda pasada: Generando aristas y calculando pesos...")
    edge_list, weight_list = [], []
    with open(user_filepath, 'r') as f:
        for line_num, line_content in enumerate(f, start=1):
            if line_num > NUM_USERS:
                break
            
            src_id = line_num
            if src_id not in id2idx or not (1 <= src_id <= len(locations_np)):
                continue

            src_idx = id2idx[src_id]
            coord_src = locations_np[src_id - 1]

            for dst_str in line_content.strip().split(','):
                cleaned_dst_str = dst_str.strip()
                if not cleaned_dst_str:
                    continue
                
                try:
                    dst_id = int(cleaned_dst_str)
                    if dst_id in id2idx and (1 <= dst_id <= len(locations_np)):
                        dst_idx = id2idx[dst_id]
                        coord_dst = locations_np[dst_id - 1]
                        
                        distance = haversine(coord_src, coord_dst)
                        edge_list.append((src_idx, dst_idx))
                        weight_list.append(distance)
                except ValueError:
                    logging.warning(f"Dato no num\u00e9rico {repr(cleaned_dst_str)} encontrado en la l\u00ednea {line_num}. Se ignora.")
                    continue

    logging.info(f"Construyendo grafo igraph con {len(edge_list):,} aristas ponderadas...")
    g = ig.Graph(n=num_unique_nodes, edges=edge_list, directed=True, edge_attrs={'weight': weight_list})
    logging.info(f"Grafo ponderado final creado con {g.vcount():,} nodos y {g.ecount():,} aristas.")
    return g, id2idx, idx2id

# =========================================================
def save_processed_data(g, id2idx, idx2id, locations_np):
    """
    Guarda todos los artefactos procesados en archivos pickle.
    Incluye: grafo, mapeos id<->idx y ubicaciones.
    """
    logging.info("Iniciando guardado de datos procesados...")
    if g:
        g.write_pickle(GRAPH_IGRAPH_FILE)
        logging.info(f"Grafo guardado en {GRAPH_IGRAPH_FILE}")
    if id2idx:
        with open(ID2IDX_PKL_FILE, 'wb') as f: pickle.dump(id2idx, f)
        logging.info(f"Mapeo id->idx guardado en {ID2IDX_PKL_FILE}")
    if idx2id:
        with open(IDX2ID_PKL_FILE, 'wb') as f: pickle.dump(idx2id, f)
        logging.info(f"Mapeo idx->id guardado en {IDX2ID_PKL_FILE}")
    if locations_np is not None and id2idx:
        locations_dict = {
            user_id: tuple(locations_np[user_id - 1])
            for user_id in id2idx.keys() if 1 <= user_id <= len(locations_np)
        }
        with open(LOCATIONS_PKL_FILE, 'wb') as f: pickle.dump(locations_dict, f)
        logging.info(f"Diccionario de ubicaciones guardado en {LOCATIONS_PKL_FILE}")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    logging.info("\n============================================\n--- INICIANDO SCRIPT DE CONSTRUCCI\u00d3N DE GRAFO PONDERADO ---\n============================================")
    start_time = time.time()
    
    locations_data = load_locations(LOCATION_TXT_FILE, NUM_USERS)
    graph_igraph, id_to_idx, idx_to_id = build_weighted_graph(USER_TXT_FILE, locations_data)
    
    if graph_igraph is not None:
        save_processed_data(graph_igraph, id_to_idx, idx_to_id, locations_data)
        end_time = time.time()
        logging.info(f"\n============================================\n--- SCRIPT DE CONSTRUCCI\u00d3N FINALIZADO EXITOSAMENTE en {end_time - start_time:.2f} segundos ---\n============================================")
    else:
        logging.critical("No se pudo construir el grafo. Proceso abortado.")