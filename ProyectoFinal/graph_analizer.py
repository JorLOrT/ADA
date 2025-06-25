import numpy as np
import igraph as ig
import time
import os
import logging
import pickle
import heapq

from collections import defaultdict
import community_louvain as community


USER_TXT_FILE = './dataset/1_million_user.txt'
OUTPUT_DIR = './processed_data' 
GRAPH_IGRAPH_FILE = os.path.join(OUTPUT_DIR, "social_network_graph.igraph.pkl")
IDX2ID_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_idx2id.pkl")
ID2IDX_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_id2idx.pkl")
LOG_FILE = os.path.join(OUTPUT_DIR,"graph_analizer.log")
SAMPLE_SIZE_FOR_PATHS = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [Analyzer] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

def load_preprocessed_data():
    logging.info("Iniciando la carga de datos pre-procesados...")
    try:
        g = ig.Graph.Read_Pickle(GRAPH_IGRAPH_FILE)
        with open(IDX2ID_PKL_FILE, 'rb') as f: idx2id = pickle.load(f)
        with open(ID2IDX_PKL_FILE, 'rb') as f: id2idx = pickle.load(f)
        logging.info("Datos de grafo y mapeos cargados.")
        return g, id2idx, idx2id
    except FileNotFoundError as e:
        logging.error(f"Error: Archivo no encontrado - {e.filename}. Ejecuta 'graph_builder.py' primero.")
        return None, None, None

def analyze_basic_metrics(g):
    logging.info("--- 1. Análisis de Métricas Básicas ---")
    logging.info(f"Nodos: {g.vcount():,}, Aristas: {g.ecount():,}, Densidad: {g.density():.4e}")

def analyze_degree_centrality(g, idx2id, top_n=10):
    logging.info(f"--- 2. Análisis de Centralidad de Grado (Top {top_n}) ---")
    if not g.is_directed():
        logging.warning("ADVERTENCIA: El grafo cargado no es dirigido. Grado de entrada y salida serán idénticos.")
    
    in_degrees = g.degree(mode="in")
    top_in_indices = np.argsort(in_degrees)[-top_n:][::-1]
    logging.info("Top Usuarios con más seguidores (Más Populares):")
    for idx in top_in_indices:
        logging.info(f"  - ID {idx2id.get(idx, 'N/A')}: {in_degrees[idx]:,} seguidores")

    out_degrees = g.degree(mode="out")
    top_out_indices = np.argsort(out_degrees)[-top_n:][::-1]
    logging.info("Top Usuarios que siguen a más personas (Más Activos):")
    for idx in top_out_indices:
        logging.info(f"  - ID {idx2id.get(idx, 'N/A')}: sigue a {out_degrees[idx]:,} usuarios")

def analyze_connectivity(g):
    logging.info("--- 3. Análisis de Conectividad (WCC y SCC) ---")
    start = time.time()
    # WCC: componentes conexas en el grafo NO dirigido
    g_undirected = g.as_undirected(combine_edges='first')
    wcc = g_undirected.components(mode='weak')
    logging.info("Componentes Débilmente Conectados (WCC):")
    logging.info(f"  - Número total de componentes: {len(wcc):,}")
    if wcc:
        giant_wcc = wcc.giant()
        logging.info(f"  - Tamaño del componente gigante: {giant_wcc.vcount():,} nodos ({giant_wcc.vcount()/g.vcount()*100:.2f}%)")
    # SCC: componentes fuertemente conexas en el grafo dirigido
    scc = g.components(mode='strong')
    logging.info("Componentes Fuertemente Conectados (SCC):")
    logging.info(f"  - Número total de componentes: {len(scc):,}")
    if scc:
        giant_scc = scc.giant()
        logging.info(f"  - Tamaño del componente gigante: {giant_scc.vcount():,} nodos ({giant_scc.vcount()/g.vcount()*100:.2f}%)")
    logging.info(f"Análisis de conectividad completado en {time.time() - start:.2f}s.")

def dijkstra(g, start_node):
    distances = defaultdict(lambda: float('inf'))
    distances[start_node] = 0
    pq = [(0, start_node)]
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > distances[current_node]: continue
        for neighbor_idx in g.successors(current_node):
            edge_id = g.get_eid(current_node, neighbor_idx)
            weight = g.es[edge_id]['weight']
            if weight == float('inf'): continue
            distance = current_dist + weight
            if distance < distances[neighbor_idx]:
                distances[neighbor_idx] = distance
                heapq.heappush(pq, (distance, neighbor_idx))
    return distances

def calculate_avg_shortest_path_sample(g, sample_size):
    logging.info(f"--- 4. Camino Más Corto Promedio Ponderado (Muestra de {sample_size}) ---")
    if 'weight' not in g.edge_attributes():
        logging.error("El grafo no está ponderado. No se puede ejecutar este análisis.")
        return
        
    start = time.time()
    total_path_length = 0
    total_paths_counted = 0
    source_nodes = np.random.choice(g.vcount(), size=min(sample_size, g.vcount()), replace=False)

    logging.info(f"Ejecutando Dijkstra ponderado desde {len(source_nodes)} nodos...")
    for i, source_node in enumerate(source_nodes):
        if (i + 1) % 1000 == 0: logging.info(f"  Procesado {i+1}/{len(source_nodes)} nodos...")
        distances = dijkstra(g, source_node)
        for dist in distances.values():
            if dist != float('inf') and dist > 0:
                total_path_length += dist
                total_paths_counted += 1
    
    avg_path = (total_path_length / total_paths_counted) if total_paths_counted > 0 else 0
    logging.info(f"Cálculo sobre muestra completado en {time.time() - start:.2f}s.")
    logging.info(f"Camino más corto promedio (estimado): {avg_path:.2f} km")

def analyze_communities(g):
    import networkx as nx
    import community_louvain as community
    logging.info("--- 5. Detección de Comunidades (Algoritmo de Louvain) ---")
    start = time.time()
    # Convertir el grafo de igraph a NetworkX
    g_undirected = g.as_undirected(combine_edges='first')
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(g_undirected.vcount()))
    for e in g_undirected.es:
        source, target = e.tuple
        if 'weight' in g_undirected.edge_attributes():
            nx_g.add_edge(source, target, weight=e['weight'])
        else:
            nx_g.add_edge(source, target)

    # Ejecutar Louvain
    partition = community.best_partition(nx_g)
    logging.info(f"Detección de comunidades completada en {time.time() - start:.2f}s.")
    comunidades = {}
    for nodo, id_comunidad in partition.items():
        comunidades.setdefault(id_comunidad, []).append(nodo)
    logging.info(f"Número de comunidades detectadas: {len(comunidades):,}")
    # Calcular modularidad si es posible
    try:
        modularidad = community.modularity(partition, nx_g)
        logging.info(f"Modularidad del grafo: {modularidad:.4f}")
    except Exception:
        pass
    # Mostrar las 5 comunidades más grandes
    top_communities = sorted(comunidades.values(), key=len, reverse=True)[:5]
    logging.info("Tamaño de las 5 comunidades más grandes:")
    for i, comm in enumerate(top_communities):
        logging.info(f"  - Comunidad {i+1}: {len(comm):,} nodos")

def kruskal_mst_implementation(g):
    logging.info("---6. Kruskal MST ---")
    import time
    start = time.time()
    g_undirected = g.as_undirected(combine_edges='first')
    num_nodes = g_undirected.vcount()
    # 1. Obtener todas las aristas y pesos
    edges = []
    for e in g_undirected.es:
        source, target = e.tuple
        weight = e['weight'] if 'weight' in g_undirected.edge_attributes() else 1.0
        edges.append((weight, source, target))
    # 2. Ordenar aristas por peso
    edges.sort()
    # 3. Estructura Union-Find
    parent = list(range(num_nodes))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
        return True
    # 4. Construir el MST
    mst_edges = []
    total_weight = 0.0
    for weight, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == num_nodes - 1:
                break
    elapsed = time.time() - start
    logging.info(f"MST Kruskal contiene {len(mst_edges):,} aristas.")
    logging.info(f"Coste total del Kruskal (distancia total): {total_weight:,.2f} km")
    logging.info(f"Tiempo de ejecución Kruskal: {elapsed:.2f} segundos")
    return mst_edges, total_weight

def prim_mst_implementation(g):
    """
    Implementación de Prim usando heapq sobre un grafo igraph no dirigido y ponderado.
    Devuelve las aristas del MST y el coste total.
    """
    logging.info("---7. Prim MST ---")
    import heapq, time
    start = time.time()
    g_undirected = g.as_undirected(combine_edges='first')
    num_nodes = g_undirected.vcount()
    if num_nodes == 0:
        return [], 0.0
    visited = [False] * num_nodes
    min_edges = []  # heap: (peso, origen, destino)
    mst_edges = []
    total_weight = 0.0
    # Comenzar desde el nodo 0
    visited[0] = True
    for e in g_undirected.incident(0):
        edge = g_undirected.es[e]
        source, target = edge.tuple
        neighbor = target if source == 0 else source
        weight = edge['weight'] if 'weight' in g_undirected.edge_attributes() else 1.0
        heapq.heappush(min_edges, (weight, 0, neighbor))
    while min_edges and len(mst_edges) < num_nodes - 1:
        weight, u, v = heapq.heappop(min_edges)
        if visited[v]:
            continue
        visited[v] = True
        mst_edges.append((u, v, weight))
        total_weight += weight
        for e in g_undirected.incident(v):
            edge = g_undirected.es[e]
            s, t = edge.tuple
            neighbor = t if s == v else s
            if not visited[neighbor]:
                w = edge['weight'] if 'weight' in g_undirected.edge_attributes() else 1.0
                heapq.heappush(min_edges, (w, v, neighbor))
    elapsed = time.time() - start
    logging.info(f"MST Prim contiene {len(mst_edges):,} aristas.")
    logging.info(f"Coste total del Prim (distancia total): {total_weight:,.2f} km")
    logging.info(f"Tiempo de ejecución Prim: {elapsed:.2f} segundos")
    return mst_edges, total_weight

def get_users_followed_by(g, target_id, id2idx, idx2id):
    logging.info(f"--- 8. Consultando a quién sigue el usuario ID {target_id} ---")
    if target_id not in id2idx:
        logging.warning(f"El usuario con ID {target_id} no fue encontrado en el grafo.")
        return []
    user_idx = id2idx[target_id]
    following_indices = g.successors(user_idx)
    following_ids = [idx2id[idx] for idx in following_indices]
    return following_ids

if __name__ == "__main__":
    logging.info("--- INICIANDO SCRIPT DE ANÁLISIS DE GRAFO ---")
    
    graph, id2idx, idx2id = load_preprocessed_data()

    if graph and id2idx and idx2id:
        analyze_basic_metrics(graph)
        analyze_degree_centrality(graph, idx2id)
        analyze_connectivity(graph)
        calculate_avg_shortest_path_sample(graph, sample_size=SAMPLE_SIZE_FOR_PATHS)
        analyze_communities(graph)
        kruskal_mst_implementation(graph)
        prim_mst_implementation(graph)
        
        target_user_id = 223
        followed_by_target = get_users_followed_by(graph, target_user_id, id2idx, idx2id)
        if followed_by_target:
            logging.info(f"El usuario con ID {target_user_id} sigue a {len(followed_by_target):,} usuarios.")
            logging.info(f"  - Lista de seguidos (primeros 20): {followed_by_target[:20]}")
        else:
            logging.info(f"El usuario con ID {target_user_id} no existe o no sigue a ningún usuario.")
        
        logging.info("--- SCRIPT DE ANÁLISIS DE GRAFO FINALIZADO ---")
    else:
        logging.critical("No se pudieron cargar los datos necesarios. Abortando.")