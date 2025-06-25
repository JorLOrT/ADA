# graph_builder.py
# VERSIÓN FINAL Y VERIFICADA: Corrige el error de parseo de la línea de usuarios.

import numpy as np
import igraph as ig
import time
import os
import logging
import pickle
from math import radians, sin, cos, sqrt, atan2

NUM_USERS = 10_000
# Asegúrate de que estos archivos y NUM_USERS coincidan con tu dataset
LOCATION_TXT_FILE = './dataset/1_million_location.txt'
USER_TXT_FILE = './dataset/1_million_user.txt'
OUTPUT_DIR = './processed_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nombres de archivo de salida
GRAPH_IGRAPH_FILE = os.path.join(OUTPUT_DIR, "social_network_graph.igraph.pkl")
LOCATIONS_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_locations.pkl")
IDX2ID_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_idx2id.pkl")
ID2IDX_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_id2idx.pkl")
LOG_FILE = os.path.join(OUTPUT_DIR,"graph_builder.log")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [Builder] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

def haversine_distance(coord1, coord2):
    R = 6371.0 
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    if (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0): return float('inf') 
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def load_locations_numpy(filepath, num_users):
    logging.info(f"Iniciando carga de ubicaciones desde {filepath}")
    locations = np.zeros((num_users, 2), dtype=np.float32)
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_users: break
                try:
                    # Esto ya estaba correcto, usa split(',')
                    lat, lon = map(float, line.strip().split(','))
                    locations[i] = [lat, lon]
                except (ValueError, IndexError):
                    locations[i] = [0.0, 0.0]
        logging.info("Carga de ubicaciones completada.")
        return locations
    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de ubicaciones no encontrado en {filepath}.")
        return None

def build_weighted_graph(user_filepath, locations_np):
    if locations_np is None: return None, None, None
    logging.info("Primera pasada: Recolectando todos los IDs únicos...")
    user_ids = set()
    with open(user_filepath, 'r') as f:
        for line_num, line_content in enumerate(f, start=1):
            if line_num > NUM_USERS: break
            user_ids.add(line_num)
            # Esto ya estaba correcto, usa split(',')
            dst_ids = {int(id_str.strip()) for id_str in line_content.strip().split(',') if id_str.strip()}
            user_ids.update(dst_ids)
    
    sorted_user_ids = sorted(list(user_ids))
    id2idx = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
    idx2id = {idx: uid for uid, idx in id2idx.items()}
    num_unique_nodes = len(id2idx)
    logging.info(f"Mapeados {num_unique_nodes:,} IDs únicos a índices.")

    logging.info("Segunda pasada: Generando aristas y calculando pesos...")
    edge_list, weight_list = [], []
    with open(user_filepath, 'r') as f:
        for line_num, line_content in enumerate(f, start=1):
            if line_num > NUM_USERS: break
            src_id = line_num
            if src_id not in id2idx: continue
            src_idx = id2idx[src_id]
            if not (1 <= src_id <= len(locations_np)): continue
            coord_src = locations_np[src_id - 1]
            
            # FIX: Usar split(',') para dividir la cadena correctamente.
            for dst_str in line_content.strip().split(','):
                cleaned_dst_str = dst_str.strip() # Limpiar espacios en blanco
                if not cleaned_dst_str: continue
                try:
                    dst_id = int(cleaned_dst_str)
                    if dst_id not in id2idx: continue
                    dst_idx = id2idx[dst_id]
                    if not (1 <= dst_id <= len(locations_np)): continue
                    coord_dst = locations_np[dst_id - 1]
                    distance = haversine_distance(coord_src, coord_dst)
                    edge_list.append((src_idx, dst_idx))
                    weight_list.append(distance)
                except ValueError:
                    logging.warning(f"ID no numérico '{cleaned_dst_str}' encontrado en la línea {line_num}. Ignorando.")
                    continue

    logging.info(f"Construyendo grafo igraph con {len(edge_list):,} aristas ponderadas...")
    g = ig.Graph(n=num_unique_nodes, edges=edge_list, directed=True, edge_attrs={'weight': weight_list})
    logging.info(f"Grafo ponderado final creado con {g.vcount():,} nodos y {g.ecount():,} aristas.")
    return g, id2idx, idx2id

def save_processed_data(g, id2idx, idx2id, locations_np):
    logging.info("Iniciando el guardado de TODOS los datos procesados...")
    if g: g.write_pickle(GRAPH_IGRAPH_FILE); logging.info(f"Grafo guardado.")
    if id2idx:
        with open(ID2IDX_PKL_FILE, 'wb') as f: pickle.dump(id2idx, f)
        logging.info(f"Mapeo id->idx guardado.")
    if idx2id:
        with open(IDX2ID_PKL_FILE, 'wb') as f: pickle.dump(idx2id, f)
        logging.info(f"Mapeo idx->id guardado.")
    if locations_np is not None and id2idx:
        locations_dict = {
            user_id: tuple(locations_np[user_id - 1])
            for user_id in id2idx.keys() if 1 <= user_id <= len(locations_np)
        }
        with open(LOCATIONS_PKL_FILE, 'wb') as f: pickle.dump(locations_dict, f)
        logging.info(f"Diccionario de ubicaciones guardado.")

if __name__ == "__main__":
    logging.info("--- INICIANDO SCRIPT DE CONSTRUCCIÓN DE GRAFO PONDERADO ---")
    locations_data = load_locations_numpy(LOCATION_TXT_FILE, NUM_USERS)
    graph_igraph, id_to_idx, idx_to_id = build_weighted_graph(USER_TXT_FILE, locations_data)
    if graph_igraph is not None:
        save_processed_data(graph_igraph, id_to_idx, idx_to_id, locations_data)
        logging.info("--- SCRIPT DE CONSTRUCCIÓN FINALIZADO EXITOSAMENTE ---")
    else:
        logging.critical("No se pudo construir el grafo. Abortando.")