import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from tqdm import tqdm
import time
import os
import random
from collections import Counter
import pickle
import logging
import community as community_louvain
import leidenalg

PROCESS_DIR = './processed_data'
GRAPH_FILE = os.path.join(PROCESS_DIR, "social_network_graph_10M.igraph.pkl") # Archivo igraph
LOCATIONS_FILE = os.path.join(PROCESS_DIR, "social_network_locations_10M.pkl")
IDX2ID_FILE = os.path.join(PROCESS_DIR, "social_network_idx2id_10M.pkl") # Necesario para nombres
ID2IDX_FILE = os.path.join(PROCESS_DIR, "social_network_id2idx_10M.pkl") # Necesario para buscar grados/centralidad por ID original
OUTPUT_VIZ_DIR = './visualizations'
LOG_FILE = "graph_visualizer.log"

# Visualization Limits (Ajustados ligeramente)
MAX_NODES_BASIC_VIZ = 300
MAX_NODES_COMMUNITY_VIZ = 1000 # Leiden puede manejar más que Louvain/NX
MAX_NODES_INTERACTIVE_VIZ = 1500 # Permitir un poco más para Plotly
MAX_NODES_GEO_VIZ = 15_000     # Aumentar un poco para el mapa

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), # Sobrescribir log en cada ejecución
                              logging.StreamHandler()])

os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

# --- Funciones ---
def load_processed_data(graph_path, id2idx_path, idx2id_path, locations_path):
    """Carga el grafo igraph, mapeos, y diccionario de ubicaciones."""
    g_igraph = None
    id2idx = None
    idx2id = None
    locations = None
    # Load igraph Graph
    logging.info(f"Cargando grafo igraph desde {graph_path}...")
    start_time = time.time()
    try:
        # Intentar con el método nativo primero
        g_igraph = ig.Graph.Read_Pickle(graph_path)
        logging.info(f"Grafo igraph cargado (Read_Pickle) en {time.time() - start_time:.2f} seg: {g_igraph.vcount()} nodos, {g_igraph.ecount()} aristas")
    except Exception as e1:
        logging.warning(f"Fallo al cargar con igraph.Read_Pickle ({e1}). Intentando con pickle.load...")
        try:
            with open(graph_path, 'rb') as f:
                g_igraph = pickle.load(f)
            logging.info(f"Grafo igraph cargado (pickle.load) en {time.time() - start_time:.2f} seg: {g_igraph.vcount()} nodos, {g_igraph.ecount()} aristas")
        except FileNotFoundError:
             logging.error(f"Error Crítico: Archivo de grafo igraph no encontrado: {graph_path}")
             return None, None, None, None
        except Exception as e2:
            logging.error(f"Error Crítico al cargar el grafo igraph con ambos métodos: {e2}", exc_info=True)
            return None, None, None, None

    # Load id2idx mapping
    logging.info(f"Cargando mapeo id->idx desde {id2idx_path}...")
    start_time = time.time()
    try:
        with open(id2idx_path, 'rb') as f:
            id2idx = pickle.load(f)
        logging.info(f"Mapeo id->idx cargado en {time.time() - start_time:.2f} seg: {len(id2idx)} entradas.")
    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de mapeo id->idx no encontrado: {id2idx_path}")
        return g_igraph, None, None, None # Devolver lo que se tenga
    except Exception as e:
        logging.error(f"Error Crítico al cargar mapeo id->idx: {e}", exc_info=True)
        return g_igraph, None, None, None

    # Load idx2id mapping
    logging.info(f"Cargando mapeo idx->id desde {idx2id_path}...")
    start_time = time.time()
    try:
        with open(idx2id_path, 'rb') as f:
            idx2id = pickle.load(f)
        logging.info(f"Mapeo idx->id cargado en {time.time() - start_time:.2f} seg: {len(idx2id)} entradas.")
    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de mapeo idx->id no encontrado: {idx2id_path}")
        return g_igraph, id2idx, None, None
    except Exception as e:
        logging.error(f"Error Crítico al cargar mapeo idx->id: {e}", exc_info=True)
        return g_igraph, id2idx, None, None

    # Load Locations
    logging.info(f"Cargando ubicaciones desde {locations_path}...")
    start_time = time.time()
    try:
        with open(locations_path, 'rb') as f:
            locations = pickle.load(f)
        logging.info(f"Ubicaciones cargadas en {time.time() - start_time:.2f} seg: {len(locations)} localizaciones.")
    except FileNotFoundError:
        logging.warning(f"Advertencia: Archivo de ubicaciones no encontrado: {locations_path}. Viz geo no funcionará.")
        locations = None
    except Exception as e:
        logging.error(f"Error al cargar las ubicaciones: {e}", exc_info=True)
        logging.warning("Continuando sin datos de ubicación debido a error.")
        locations = None

    # Validación de consistencia rápida (opcional)
    if g_igraph and id2idx and idx2id:
         if g_igraph.vcount() != len(id2idx) or g_igraph.vcount() != len(idx2id):
             logging.warning("Inconsistencia detectada: Número de nodos igraph no coincide con tamaño de mapeos.")
         sample_indices = random.sample(range(g_igraph.vcount()), min(100, g_igraph.vcount()))
         if not all(idx in idx2id for idx in sample_indices):
              logging.warning("Advertencia: Algunos índices de igraph no se encuentran en idx2id.")

    return g_igraph, id2idx, idx2id, locations

def create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes, sampling_method='random'):
    """
    Crea un subgrafo igraph muestreado y lo convierte a NetworkX.
    Devuelve el subgrafo NetworkX y los IDs originales en el subgrafo.
    """
    num_total_nodes = g_igraph.vcount()
    if num_total_nodes == 0:
        return nx.DiGraph(), []

    sampled_indices = []
    node_indices = list(range(num_total_nodes)) # Lista de todos los índices válidos

    if num_total_nodes <= max_nodes:
        sampled_indices = node_indices
        logging.info(f"Usando grafo completo ({num_total_nodes} nodos) para conversión a NetworkX.")
    else:
        logging.warning(f"Grafo igraph ({num_total_nodes} nodos) excede límite ({max_nodes}). Muestreando...")
        if sampling_method == 'random':
            # Asegurar que muestreamos de índices válidos
            sampled_indices = random.sample(node_indices, max_nodes)
        elif sampling_method == 'degree':
            logging.info("Muestreando por grado descendente...")
            try:
                degrees = g_igraph.degree(node_indices) # Grado total
                nodes_with_degree = sorted(node_indices, key=lambda i: degrees[i], reverse=True)
                sampled_indices = nodes_with_degree[:max_nodes]
            except Exception as e:
                 logging.error(f"Error muestreando por grado: {e}. Usando muestreo aleatorio.")
                 sampled_indices = random.sample(node_indices, max_nodes)
        else:
            logging.error(f"Método de muestreo '{sampling_method}' no implementado. Usando 'random'.")
            sampled_indices = random.sample(node_indices, max_nodes)
        logging.info(f"Muestreo completado, {len(sampled_indices)} índices seleccionados.")

    if not sampled_indices:
         logging.warning("No se seleccionaron índices para el subgrafo.")
         return nx.DiGraph(), []

    # Crear subgrafo igraph
    try:
        subg_igraph = g_igraph.subgraph(sampled_indices)
    except Exception as e:
        logging.error(f"Error al crear subgrafo igraph: {e}. Indices: {sampled_indices[:10]}...")
        return nx.DiGraph(), []

    # Obtener IDs originales (asegurarse que el índice existe en idx2id)
    subgraph_original_ids = [idx2id[i] for i in sampled_indices if i in idx2id]
    if len(subgraph_original_ids) != len(sampled_indices):
         logging.warning("Discrepancia entre índices muestreados y mapeo idx2id.")

    # Convertir SOLO el subgrafo a NetworkX
    logging.info(f"Convirtiendo subgrafo igraph de {subg_igraph.vcount()} nodos a NetworkX...")
    start_conv = time.time()
    subG_nx = None
    try:
        # Asignar IDs originales como nombres de vértices ANTES de convertir
        # Usar str() para asegurar compatibilidad si IDs son numéricos
        subg_igraph.vs["name"] = [str(idx2id[v.index]) for v in subg_igraph.vs if v.index in idx2id]
        # Si no todos los vs.index están en idx2id, esto fallará o dará nombres incorrectos.
        # Necesitamos asegurar que vs.index se refiere al índice original del grafo completo.
        # La forma correcta es iterar sobre los índices muestreados:
        vertex_names = {v_idx: str(idx2id[v_idx]) for v_idx in sampled_indices if v_idx in idx2id}
        # Mapear los índices del *subgrafo* a los nombres correctos
        # Necesitamos mapear índice_subgrafo -> índice_original -> nombre
        original_indices_in_subgraph = [v.index for v in subg_igraph.vs] # Índices originales de los nodos que QUEDARON en el subgrafo
        subg_igraph.vs["name"] = [vertex_names[orig_idx] for orig_idx in original_indices_in_subgraph]


        # Convertir usando 'name' como ID de nodo NX
        subG_nx = subg_igraph.to_networkx(vertex_attr_hashable="name")

        # Convertir los nombres (que son strings) de nuevo a int si eran originalmente ints
        id_map_nx = {name: int(name) for name in subG_nx.nodes()}
        subG_nx = nx.relabel_nodes(subG_nx, id_map_nx, copy=True)


        elapsed_conv = time.time() - start_conv
        logging.info(f"Subgrafo convertido a NetworkX en {elapsed_conv:.2f} segundos ({subG_nx.number_of_nodes()} nodos, {subG_nx.number_of_edges()} aristas).")

    except KeyError as e:
         logging.error(f"Error de clave (KeyError) convirtiendo a NetworkX. Índice {e} no encontrado en idx2id. Asegúrate que los mapeos son correctos.")
         return nx.DiGraph(), []
    except Exception as e:
        logging.error(f"Error convirtiendo subgrafo igraph a NetworkX: {e}", exc_info=True)
        return nx.DiGraph(), []

    return subG_nx, subgraph_original_ids


# --- Funciones de Cálculo Auxiliares ---

def get_original_degree(node_id, g_igraph, id2idx, mode='in'):
    """Obtiene el grado del nodo del grafo igraph original."""
    try:
        idx = id2idx.get(node_id) # Mapear ID original a índice igraph
        if idx is not None and idx < g_igraph.vcount(): # Validar índice
            igraph_mode = ig.IN if mode=='in' else (ig.OUT if mode=='out' else ig.ALL)
            return g_igraph.degree(idx, mode=igraph_mode)
        return 0 # Si no se encuentra el ID o índice inválido
    except Exception as e:
        return 0

def calculate_centralities(g_igraph, idx2id, measure='pagerank'):
    """Calcula una medida de centralidad en el grafo igraph completo."""
    if not g_igraph: return None
    num_nodes = g_igraph.vcount()
    if num_nodes == 0: return {}

    logging.info(f"Calculando centralidad '{measure}' en grafo igraph completo ({num_nodes} nodos)...")
    start_time = time.time()
    centrality_values = {}
    raw_values = []

    try:
        if measure == 'pagerank':
            raw_values = g_igraph.pagerank(implementation="prpack")

        elif measure == 'betweenness':
            # Considera añadir un límite o usar una aproximación si es necesario.
            logging.warning("El cálculo de Betweenness Centrality en el grafo completo puede ser extremadamente lento o fallar por memoria.")
            raw_values = g_igraph.betweenness(directed=True)

        elif measure == 'eigenvector':
            logging.info("Calculando Eigenvector Centrality (puede ser lento)...")
            raw_values = g_igraph.eigenvector_centrality(directed=True, scale=True)

        else:
            logging.error(f"Medida de centralidad '{measure}' no reconocida.")
            return None

        # Mapear los resultados (indexados por igraph) a IDs originales
        # Iterar sobre los índices válidos del grafo (0 a vcount-1)
        for i in range(num_nodes):
            if i < len(raw_values) and i in idx2id:
                 centrality_values[idx2id[i]] = raw_values[i]
            # else: # Loggear si falta mapeo o valor (debería ser raro)
            #    logging.debug(f"Índice igraph {i} sin mapeo o valor de centralidad.")


        elapsed = time.time() - start_time
        logging.info(f"Cálculo de centralidad '{measure}' completado en {elapsed:.2f} seg.")
        return centrality_values

    except MemoryError:
         logging.error(f"Error de Memoria calculando centralidad '{measure}'.")
         return None
    except Exception as e:
        logging.error(f"Error calculando centralidad '{measure}': {e}", exc_info=True)
        return None

# --- Funciones de Visualización ---

def basic_graph_visualization(g_igraph, idx2id, title="Subgrafo Red Social (Estático)", save_path=None, max_nodes=MAX_NODES_BASIC_VIZ):
    """Visualiza un subgrafo pequeño usando matplotlib, convirtiendo solo el subgrafo."""
    if g_igraph is None or not idx2id:
        logging.error("Datos de grafo igraph o mapeo no disponibles para viz básica.")
        return

    # Crear subgrafo y convertirlo a NetworkX
    # Usar muestreo por grado para intentar capturar nodos más interesantes
    subG_nx, _ = create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes, sampling_method='degree')
    if subG_nx.number_of_nodes() == 0:
         subG_nx, _ = create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes, sampling_method='random')

    if subG_nx.number_of_nodes() == 0:
        logging.warning("Subgrafo NetworkX para visualización básica está vacío.")
        return

    plt.figure(figsize=(12, 12)) # Un poco más grande
    logging.info(f"Calculando layout (spring) para {subG_nx.number_of_nodes()} nodos...")
    try:
        # Ajustar k basado en el número de nodos para evitar superposición
        k_val = 0.8 / np.sqrt(subG_nx.number_of_nodes()) if subG_nx.number_of_nodes() > 1 else 0.8
        pos = nx.spring_layout(subG_nx, seed=42, k=k_val, iterations=50) # Más iteraciones
    except Exception as e:
        logging.error(f"Error calculando layout spring: {e}. Usando random layout.")
        pos = nx.random_layout(subG_nx, seed=42)

    # Colorear por grado de entrada, tamaño por grado de salida (con logs)
    try:
        # Usar grados del subgrafo para la visualización local
        in_degree_sub = dict(subG_nx.in_degree())
        out_degree_sub = dict(subG_nx.out_degree())
        node_color = [np.log1p(in_degree_sub.get(node, 0)) for node in subG_nx.nodes()]
        node_size = [20 + 15 * np.log1p(out_degree_sub.get(node, 0)) for node in subG_nx.nodes()]
    except Exception as e:
        logging.error(f"Error calculando grados del subgrafo para colorear/tamaño: {e}")
        node_color = 'skyblue'; node_size = 30

    logging.info("Dibujando grafo con Matplotlib...")
    nx.draw(
        subG_nx, pos=pos, node_color=node_color, node_size=node_size, cmap=plt.cm.viridis,
        alpha=0.8, with_labels=False, arrows=True, arrowsize=10, # Flechas un poco más grandes
        edge_color='#cccccc', # Gris más claro
        width=0.4 # Ancho de línea
        )

    plt.title(f"{title}\n(Muestra de {subG_nx.number_of_nodes()} nodos / {subG_nx.number_of_edges()} aristas)", fontsize=14)
    plt.axis('off')

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=200, bbox_inches='tight') # Mayor DPI
            logging.info(f"Visualización básica guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización básica: {e}")
    else: plt.show()
    plt.close() # Liberar memoria


def community_visualization(g_igraph, id2idx, idx2id, save_path=None, max_nodes=MAX_NODES_COMMUNITY_VIZ):
    """Detecta comunidades (Leiden preferido) y visualiza estáticamente."""
    if g_igraph is None or not id2idx or not idx2id:
        logging.error("Datos de grafo igraph o mapeos no disponibles para viz de comunidades.")
        return None
    if leidenalg is None and community_louvain is None:
         logging.error("No hay algoritmos de detección de comunidad disponibles (leidenalg o python-louvain).")
         return None

    partition = None
    num_communities = 0
    modularity = None
    algo_used = "None"

    try:
        num_total_nodes = g_igraph.vcount()
        node_indices = list(range(num_total_nodes))
        sampled_indices_comm = node_indices if num_total_nodes <= max_nodes else random.sample(node_indices, max_nodes)

        if not sampled_indices_comm:
            logging.warning("No se seleccionaron nodos para detección de comunidades.")
            return None

        subg_igraph_comm = g_igraph.subgraph(sampled_indices_comm)
        logging.info(f"Detectando comunidades en subgrafo igraph de {subg_igraph_comm.vcount()} nodos...")

        if leidenalg:
             algo_used = "Leiden"
             logging.info("Usando algoritmo Leiden (igraph)...")
             part = leidenalg.find_partition(subg_igraph_comm.copy().to_undirected(), leidenalg.ModularityVertexPartition, seed=42)
             membership = part.membership
             modularity = part.modularity
             # Mapear resultado a IDs originales
             partition = {idx2id[subg_igraph_comm.vs[i].index]: membership[i] for i in range(subg_igraph_comm.vcount())}
             num_communities = len(part)
             logging.info(f"Leiden completado. {num_communities} comunidades, Modularidad: {modularity:.4f}")
        elif community_louvain:
             algo_used = "Louvain"
             logging.warning("Usando Louvain (requiere conversión a NetworkX)...")
             # Convertir subgrafo a NetworkX para Louvain
             subG_nx_comm, _ = create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes) # Usar el mismo límite
             if subG_nx_comm.number_of_nodes() > 0:
                  undirected_G = subG_nx_comm.to_undirected()
                  # Louvain puede ser sensible a nodos aislados
                  undirected_G.remove_nodes_from(list(nx.isolates(undirected_G)))
                  if undirected_G.number_of_nodes() > 0:
                       logging.info(f"Detectando (Louvain en NX) en {undirected_G.number_of_nodes()} nodos...")
                       start_time = time.time()
                       partition = community_louvain.best_partition(undirected_G, random_state=42)
                       # Calcular modularidad para la partición en el grafo donde se calculó
                       modularity = community_louvain.modularity(partition, undirected_G)
                       elapsed_time = time.time() - start_time
                       num_communities = len(set(partition.values()))
                       logging.info(f"Louvain (NX) completado en {elapsed_time:.2f} seg. {num_communities} com., Mod: {modularity:.4f}")
                  else: logging.warning("Subgrafo NX sin nodos no aislados para Louvain.")
             else: logging.warning("Subgrafo NX vacío para Louvain.")
        else:
            # No debería llegar aquí por el check inicial, pero por si acaso
            logging.error("No se encontró método de detección de comunidad.")
            return None

    except Exception as e:
        logging.error(f"Error durante la detección de comunidades ({algo_used}): {e}", exc_info=True)
        return None # Retornar None si la detección falla

    if not partition:
        logging.warning("No se pudo generar la partición de comunidades.")
        return None # Retornar None si no hay partición

    # --- Visualización Estática (Matplotlib con NetworkX) ---
    subG_nx_viz, _ = create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes)

    if subG_nx_viz.number_of_nodes() == 0:
         logging.warning("Subgrafo NX para visualización de comunidades está vacío.")
         return partition # Devolver partición aunque no se visualice

    nodes_to_draw = list(subG_nx_viz.nodes())
    # Filtrar partición para incluir solo nodos que están en el grafo a dibujar
    valid_partition_for_viz = {node: comm for node, comm in partition.items() if node in nodes_to_draw}
    if len(valid_partition_for_viz) < len(nodes_to_draw):
         logging.warning("Algunos nodos del subgrafo de visualización no tienen ID de comunidad.")

    plt.figure(figsize=(14, 14)) # Más grande y cuadrado
    logging.info("Calculando layout para visualización de comunidades (NetworkX)...")
    G_layout = subG_nx_viz.to_undirected() # Layout suele verse mejor en no dirigido
    try:
        k_val = 0.9 / np.sqrt(G_layout.number_of_nodes()) if G_layout.number_of_nodes() > 1 else 0.9
        pos = nx.spring_layout(G_layout, seed=42, k=k_val, iterations=60) # Más iteraciones
    except Exception as e:
        logging.error(f"Error calculando layout: {e}. Usando random layout."); pos = nx.random_layout(G_layout, seed=42)

    # Mapear comunidades a colores
    unique_communities = sorted(list(set(valid_partition_for_viz.values())))
    num_actual_communities = len(unique_communities)
    # Usar un colormap adecuado para categorías, asegurar suficientes colores
    colors = plt.cm.get_cmap('turbo', num_actual_communities) if num_actual_communities > 20 else plt.cm.get_cmap('tab20', num_actual_communities)
    community_to_color = {comm_id: colors(i) for i, comm_id in enumerate(unique_communities)}
    # Asignar color, usar un color por defecto (gris) si falta comunidad
    default_color = (0.8, 0.8, 0.8, 0.5) # RGBA gris claro semi-transparente
    node_colors = [community_to_color.get(valid_partition_for_viz.get(node), default_color) for node in G_layout.nodes()]

    logging.info("Dibujando nodos y aristas de comunidades...")
    nx.draw_networkx_nodes(G_layout, pos, node_color=node_colors, node_size=35, alpha=0.85)
    nx.draw_networkx_edges(G_layout, pos, edge_color='#dddddd', width=0.25, alpha=0.6)

    mod_text = f"Modularidad: {modularity:.4f}" if modularity is not None else "Modularidad: N/A"
    plt.title(f"Comunidades en Subgrafo ({G_layout.number_of_nodes()} nodos) - Algoritmo: {algo_used}\n"
              f"{num_communities} comunidades detectadas | {mod_text}", fontsize=15)
    plt.axis('off')

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=200, bbox_inches='tight')
            logging.info(f"Visualización de comunidades guardada en {full_save_path}")
        except Exception as e: logging.error(f"Error al guardar viz de comunidades: {e}")
    else: plt.show()
    plt.close()

    return partition # Devolver la partición completa calculada


def interactive_visualization(g_igraph, id2idx, idx2id, locations=None, communities=None,
                              centrality_map=None, centrality_name='Centrality',
                              title="Red Social Interactiva", save_path=None, max_nodes=MAX_NODES_INTERACTIVE_VIZ):
    """Crea una visualización interactiva (Plotly) de un subgrafo, con opciones de color/tamaño."""
    if g_igraph is None or not id2idx or not idx2id:
        logging.error("Datos de grafo igraph o mapeos no disponibles para viz interactiva.")
        return None

    # Crear subgrafo y convertirlo a NetworkX
    # Muestrear aleatoriamente para no sesgar por grado si coloreamos por centralidad
    subG_nx, subgraph_original_ids = create_sampled_networkx_subgraph(g_igraph, idx2id, max_nodes, sampling_method='random')

    if subG_nx.number_of_nodes() == 0:
        logging.warning("Subgrafo NetworkX para visualización interactiva está vacío.")
        return None

    logging.info(f"Preparando visualización interactiva para {subG_nx.number_of_nodes()} nodos...")

    # --- Layout (Geográfico o Spring 3D) ---
    pos_3d = {}
    use_geo_layout = False
    subG_nx_viz = subG_nx # Usar una copia por si filtramos para geo
    if locations:
        nodes_with_loc = {n for n in subG_nx.nodes() if n in locations}
        coverage = len(nodes_with_loc) / subG_nx.number_of_nodes() if subG_nx.number_of_nodes() > 0 else 0
        logging.info(f"Cobertura de ubicaciones en subgrafo NX: {coverage*100:.1f}%")
        if coverage > 0.7:
             logging.info("Usando coordenadas geográficas para layout 3D.")
             use_geo_layout = True
             subG_nodes_with_loc = list(nodes_with_loc)
             subG_nx_filtered = subG_nx.subgraph(subG_nodes_with_loc).copy()
             if subG_nx_filtered.number_of_nodes() == 0: logging.error("Ningún nodo tenía ubicación."); return None
             logging.info(f"Filtrando subgrafo NX a {subG_nx_filtered.number_of_nodes()} nodos con ubicación.")
             for node in tqdm(subG_nx_filtered.nodes(), desc="Calculando coords 3D desde Lat/Lon"):
                 lat, lon = locations[node]
                 lat_rad, lon_rad = np.radians(lat), np.radians(lon)
                 x = np.cos(lat_rad) * np.cos(lon_rad); y = np.cos(lat_rad) * np.sin(lon_rad); z = np.sin(lat_rad)
                 pos_3d[node] = (x, y, z)
             subG_nx_viz = subG_nx_filtered
        else: logging.info("Cobertura de ubicación insuficiente. Usando layout spring 3D.")

    if not use_geo_layout:
        logging.info("Calculando layout 3D (spring)...")
        try:
             if subG_nx_viz.number_of_nodes() > 0:
                  k_val = 0.6 / np.sqrt(subG_nx_viz.number_of_nodes()) if subG_nx_viz.number_of_nodes() > 1 else 0.6
                  pos_3d = nx.spring_layout(subG_nx_viz, seed=42, dim=3, k=k_val, iterations=50)
             else: logging.warning("Subgrafo vacío antes de layout spring 3D."); return None
        except Exception as e: logging.error(f"Error layout 3D spring: {e}. Cancelando."); return None


    # --- Preparar Datos Plotly ---
    # Edge Trace (usando subG_nx_viz y pos_3d)
    edge_x, edge_y, edge_z = [], [], []
    logging.info("Preparando datos de aristas para Plotly...")
    for src, tgt in tqdm(subG_nx_viz.edges(), desc="Procesando aristas"):
        if src in pos_3d and tgt in pos_3d:
            x0, y0, z0 = pos_3d[src]; x1, y1, z1 = pos_3d[tgt]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                              line=dict(color='rgba(200,200,200,0.3)', width=1.5), hoverinfo='none')

    # Node Trace Data
    node_x = [pos_3d[node][0] for node in subG_nx_viz.nodes()]
    node_y = [pos_3d[node][1] for node in subG_nx_viz.nodes()]
    node_z = [pos_3d[node][2] for node in subG_nx_viz.nodes()]
    node_texts = []
    node_colors = []
    color_title = 'Node Info'
    colorscale = 'Viridis'
    node_sizes = [5] * subG_nx_viz.number_of_nodes() # Default size

    # Determinar Color/Tamaño basado en prioridad: Centralidad > Comunidad > Grado
    coloring_mode = "Grado" # Por defecto
    if centrality_map:
        nodes_with_centrality = {n for n in subG_nx_viz.nodes() if n in centrality_map}
        if len(nodes_with_centrality) / subG_nx_viz.number_of_nodes() > 0.7:
             logging.info(f"Prioridad de color/tamaño: '{centrality_name}'.")
             coloring_mode = centrality_name
             # Escalar valores para color (log) y tamaño (lineal normalizado)
             valid_centralities = [centrality_map[n] for n in nodes_with_centrality if centrality_map[n] > 0] # Ignorar <= 0 para log/max
             max_cent = max(valid_centralities) if valid_centralities else 1
             node_colors = [np.log1p(centrality_map.get(node, 0)) for node in subG_nx_viz.nodes()]
             # Normalizar tamaño entre ~5 y ~20
             node_sizes = [5 + 15 * (centrality_map.get(node, 0) / max_cent) for node in subG_nx_viz.nodes()]
             color_title = f'Log({centrality_name})'
             colorscale = 'Plasma' # Escala buena para métricas continuas
        else: logging.warning(f"Cobertura de centralidad baja. Revisando comunidad/grado.")

    if coloring_mode == "Grado" and communities: # Si no se usó centralidad, intentar comunidad
         nodes_with_community = {n for n in subG_nx_viz.nodes() if n in communities}
         if len(nodes_with_community) / subG_nx_viz.number_of_nodes() > 0.7:
             logging.info("Prioridad de color: Comunidad.")
             coloring_mode = "Comunidad"
             community_ids = [communities.get(node, -1) for node in subG_nx_viz.nodes()]
             num_comms = len(set(community_ids) - {-1})
             node_colors = community_ids
             color_title = 'Comunidad ID'
             colorscale = 'Turbo' if num_comms > 10 else 'Viridis' # Escalas categóricas
             # Mantener tamaño por grado si coloreamos por comunidad
             node_sizes = [5 + 3 * np.log1p(get_original_degree(node, 'all', g_igraph, id2idx)) for node in subG_nx_viz.nodes()]
         else: logging.warning("Cobertura de comunidad baja. Usando grado.")

    if coloring_mode == "Grado": # Si no se usó centralidad ni comunidad
        logging.info("Prioridad de color/tamaño: Grado.")
        node_colors = [get_original_degree(node, 'in', g_igraph, id2idx) for node in subG_nx_viz.nodes()]
        color_title = 'Grado Entrada (Seguidores)'
        colorscale = 'YlGnBu'
        node_sizes = [5 + 3 * np.log1p(get_original_degree(node, 'all', g_igraph, id2idx)) for node in subG_nx_viz.nodes()]

    # Generar Hover Text
    logging.info("Generando texto de hover...")
    for i, node in enumerate(subG_nx_viz.nodes()):
        in_deg = get_original_degree(node, 'in', g_igraph, id2idx)
        out_deg = get_original_degree(node, 'out', g_igraph, id2idx)
        hover_text = f'<b>Usuario: {node}</b><br>'
        hover_text += f'Seguidores: {in_deg}<br>Siguiendo: {out_deg}'
        if communities and node in communities:
            hover_text += f'<br>Comunidad: {communities[node]}'
        if centrality_map and node in centrality_map:
            hover_text += f'<br>{centrality_name}: {centrality_map[node]:.4g}'
        if locations and node in locations:
             lat, lon = locations[node]
             hover_text += f'<br>Loc: ({lat:.3f}, {lon:.3f})'
        node_texts.append(hover_text)

    # Crear Node Trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers', name='Usuarios',
        marker=dict(symbol='circle',
                    size=node_sizes,
                    sizemode='diameter',
                    color=node_colors,
                    colorscale=colorscale,
                    colorbar=dict(title=color_title, thickness=15, len=0.7, x=1.05), # Ajustar colorbar
                    line=dict(width=0.5, color='rgba(50,50,50,0.8)'), # Borde ligero
                    opacity=0.9
                   ),
        text=node_texts,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>' # Formato de hover limpio
    )

    # Crear Figura
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'{title} ({subG_nx_viz.number_of_nodes()} N, {subG_nx_viz.number_of_edges()} E) - Color: {coloring_mode}',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=10,l=10,r=10,t=50), # Ajustar márgenes
                    scene=dict(xaxis=dict(visible=False), # Ocultar ejes completamente
                               yaxis=dict(visible=False),
                               zaxis=dict(visible=False),
                               bgcolor='rgb(245, 245, 245)', # Fondo gris claro
                               # Ajustar cámara inicial si es necesario
                               camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
                              ),
                    uirevision='constant' # Mantener estado de UI
                    )
                )

    logging.info("Visualización interactiva preparada.")
    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            fig.write_html(full_save_path, include_plotlyjs='cdn') # Usar CDN para reducir tamaño de archivo
            logging.info(f"Visualización interactiva guardada en {full_save_path}")
        except Exception as e: logging.error(f"Error al guardar la viz interactiva: {e}")

    return fig


def degree_distribution_visualization(g_igraph, save_path=None):
    """Visualiza la distribución de grados (in y out) usando igraph."""
    if g_igraph is None: logging.error("Grafo igraph no disponible."); return
    if g_igraph.vcount() == 0: logging.warning("Grafo igraph vacío."); return

    logging.info("Calculando distribuciones de grado (igraph)...")
    start_time = time.time()
    try:
        in_degrees = g_igraph.degree(mode=ig.IN)
        out_degrees = g_igraph.degree(mode=ig.OUT)
        if not in_degrees or not out_degrees: logging.warning("Grados cero."); return
        in_degree_counts = Counter(d for d in in_degrees if d > 0)
        out_degree_counts = Counter(d for d in out_degrees if d > 0)
        logging.info(f"Cálculo de grados (igraph) completado en {time.time() - start_time:.2f} segundos.")
    except MemoryError: logging.error("Error de memoria calculando grados (igraph)."); return
    except Exception as e: logging.error(f"Error calculando grados (igraph): {e}", exc_info=True); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot In-degree
    if in_degree_counts:
        in_deg, in_cnt = zip(*sorted(in_degree_counts.items()))
        ax1.loglog(in_deg, in_cnt, 'o', markersize=4, alpha=0.7, color='#1f77b4', label='Grado Entrada') # Azul estándar
        ax1.set_ylabel('Número de Nodos P(k)', fontsize=12)
    else: ax1.text(0.5, 0.5, 'No hay datos (grado > 0)', ha='center', va='center')
    ax1.set_title('Distribución Grado Entrada (Log-Log)', fontsize=14)
    ax1.set_xlabel('Grado Entrada (k)', fontsize=12)
    ax1.grid(True, which="both", ls=":", linewidth=0.6, alpha=0.7) # Estilo de rejilla
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.legend()

    # Plot Out-degree
    if out_degree_counts:
        out_deg, out_cnt = zip(*sorted(out_degree_counts.items()))
        ax2.loglog(out_deg, out_cnt, 'o', markersize=4, alpha=0.7, color='#ff7f0e', label='Grado Salida') # Naranja estándar
    else: ax2.text(0.5, 0.5, 'No hay datos (grado > 0)', ha='center', va='center')
    ax2.set_title('Distribución Grado Salida (Log-Log)', fontsize=14)
    ax2.set_xlabel('Grado Salida (k)', fontsize=12)
    ax2.grid(True, which="both", ls=":", linewidth=0.6, alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.legend()

    plt.tight_layout(pad=2.0) # Añadir padding

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=200, bbox_inches='tight')
            logging.info(f"Viz de grados guardada en {full_save_path}")
        except Exception as e: logging.error(f"Error al guardar viz de grados: {e}")
    else: plt.show()
    plt.close()


def geo_visualization(g_igraph, id2idx, idx2id, locations,
                      centrality_map=None, centrality_name='Centrality',
                      save_path=None, max_nodes=MAX_NODES_GEO_VIZ,
                      color_metric='followers'): # 'followers' o 'centrality'
    """Visualiza la distribución geográfica, opcionalmente coloreada por centralidad."""
    if g_igraph is None or not id2idx or not idx2id or locations is None:
        logging.error("Datos insuficientes para visualización geográfica.")
        return None

    logging.info("Preparando datos para visualización geográfica...")
    nodes_with_loc_in_graph = {
        orig_id for orig_id, idx in id2idx.items() if orig_id in locations
    }
    if not nodes_with_loc_in_graph: logging.warning("Ningún nodo tiene ubicación válida."); return None

    # Muestreo
    sampled_original_ids = list(nodes_with_loc_in_graph)
    if len(nodes_with_loc_in_graph) > max_nodes:
        logging.warning(f"Muestreando {max_nodes} de {len(nodes_with_loc_in_graph)} nodos con ubicación.")
        sampled_original_ids = random.sample(sampled_original_ids, max_nodes)

    if not sampled_original_ids: logging.warning("No quedan nodos tras muestreo geo."); return None

    # Crear DataFrame
    node_data = []
    logging.info(f"Extrayendo datos para {len(sampled_original_ids)} nodos del mapa...")
    for node_id in tqdm(sampled_original_ids, desc="Procesando nodos para mapa"):
        try:
            lat, lon = locations[node_id]
            # Validar coordenadas aquí también
            if isinstance(lat, (int, float, np.number)) and isinstance(lon, (int, float, np.number)) and -90 <= lat <= 90 and -180 <= lon <= 180:
                 idx = id2idx.get(node_id)
                 in_deg = g_igraph.degree(idx, mode=ig.IN) if idx is not None else 0
                 size = max(1, 5 * np.log1p(in_deg)) # Tamaño basado en seguidores (log)
                 data_entry = {
                     'id': node_id, 'latitude': float(lat), 'longitude': float(lon), # Convertir a float estándar
                     'followers': in_deg, 'viz_size': size
                 }
                 if centrality_map and node_id in centrality_map:
                     data_entry['centrality'] = centrality_map[node_id]
                 node_data.append(data_entry)
        except Exception as e: logging.error(f"Error procesando datos geo para nodo {node_id}: {e}")

    if not node_data: logging.error("No se extrajeron datos válidos para el mapa."); return None

    node_df = pd.DataFrame(node_data)
    logging.info(f"DataFrame creado con {len(node_df)} puntos válidos.")

    # Determinar color y hover data
    color_col = 'followers'
    color_scale_used = px.colors.sequential.Plasma
    hover_cols = ['followers', 'latitude', 'longitude']
    title_metric = "Seguidores"

    if color_metric == 'centrality' and 'centrality' in node_df.columns and centrality_map:
        logging.info(f"Coloreando mapa por '{centrality_name}'.")
        color_col = 'centrality'
        color_scale_used = px.colors.sequential.Viridis # Otra escala
        hover_cols.append('centrality')
        title_metric = centrality_name
        # Podríamos querer escalar logarítmicamente el color de centralidad también
        # node_df['log_centrality'] = np.log1p(node_df['centrality'])
        # color_col = 'log_centrality'
    else:
         logging.info("Coloreando mapa por número de seguidores.")


    # Crear figura Plotly Express
    logging.info("Generando mapa interactivo con Plotly Express...")
    fig = None
    try:
        fig = px.scatter_mapbox(
                           node_df,
                           lat='latitude',
                           lon='longitude',
                           color=color_col,
                           size='viz_size', # Usar el tamaño precalculado
                           hover_name='id',
                           hover_data={col: True for col in hover_cols}, # Mostrar columnas especificadas
                           # projection='natural earth', # No aplica a mapbox
                           title=f'Distribución Geográfica ({len(node_df)} usuarios) - Color: {title_metric}',
                           color_continuous_scale=color_scale_used,
                           size_max=18, # Tamaño máximo del punto
                           zoom=1, # Zoom inicial
                           opacity=0.7,
                           mapbox_style="carto-positron" # Estilo de mapa base limpio
                           )

        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
                          coloraxis_colorbar=dict(title=title_metric)) # Título de la barra de color

        logging.info("Mapa geográfico generado.")
    except Exception as e:
        logging.error(f"Error al generar el mapa geográfico con Plotly Express: {e}", exc_info=True)
        return None

    if fig and save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            fig.write_html(full_save_path, include_plotlyjs='cdn')
            logging.info(f"Viz geo guardada en {full_save_path}")
        except Exception as e: logging.error(f"Error al guardar viz geo: {e}")
    return fig

def visualize_graph_relationships(g_igraph, id2idx, idx2id, max_nodes=1000, save_path=None):
    """
    Crea una visualización interactiva 3D de las relaciones del grafo con flechas direccionales.
    
    Args:
        g_igraph: Grafo de igraph
        id2idx: Diccionario de mapeo ID -> índice
        idx2id: Diccionario de mapeo índice -> ID
        max_nodes: Número máximo de nodos a visualizar
        save_path: Ruta donde guardar el HTML resultante
    
    Returns:
        figura de Plotly
    """
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np
    from tqdm import tqdm
    import random
    import os
    
    # Verificar datos de entrada
    if g_igraph is None or not id2idx or not idx2id:
        logging.error("Datos de grafo o mapeos no disponibles")
        return None
    
    try:
        # Seleccionar nodos aleatoriamente
        available_ids = list(id2idx.keys())
        if len(available_ids) > max_nodes:
            selected_ids = random.sample(available_ids, max_nodes)
        else:
            selected_ids = available_ids
        
        # Crear subgrafo NetworkX
        subG_nx = nx.DiGraph()
        subG_nx.add_nodes_from(selected_ids)
        
        # Añadir aristas verificando conexiones válidas
        for orig_id in tqdm(selected_ids, desc="Procesando conexiones"):
            idx = id2idx[orig_id]
            neighbors = g_igraph.neighbors(idx, mode='out')
            for neighbor_idx in neighbors:
                if neighbor_idx in idx2id:
                    neighbor_id = idx2id[neighbor_idx]
                    if neighbor_id in subG_nx:
                        subG_nx.add_edge(orig_id, neighbor_id)
        
        # Información de diagnóstico
        logging.info(f"""
        Diagnóstico del subgrafo:
        - Nodos: {subG_nx.number_of_nodes()}
        - Aristas: {subG_nx.number_of_edges()}
        - Densidad: {nx.density(subG_nx):.4f}
        - Nodos aislados: {len(list(nx.isolates(subG_nx)))}
        """)
        
        # Calcular layout 3D
        logging.info("Calculando layout 3D...")
        pos_3d = nx.spring_layout(subG_nx, dim=3, k=1/np.sqrt(subG_nx.number_of_nodes()))
        
        # Crear trazas para aristas con flechas
        edge_traces = []
        for edge in subG_nx.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            
            # Línea principal de la conexión
            edge_trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(
                    color='rgba(200,200,200,0.5)',
                    width=1
                ),
                hoverinfo='none'
            )
            
            # Calcular punto para la flecha
            arrow_x = x0 + 0.8*(x1-x0)
            arrow_y = y0 + 0.8*(y1-y0)
            arrow_z = z0 + 0.8*(z1-z0)
            
            # Punta de la flecha
            arrow_trace = go.Scatter3d(
                x=[arrow_x, x1],
                y=[arrow_y, y1],
                z=[arrow_z, z1],
                mode='lines',
                line=dict(
                    color='rgba(200,50,50,0.8)',
                    width=2
                ),
                hoverinfo='none'
            )
            
            edge_traces.extend([edge_trace, arrow_trace])
        
        # Preparar datos para los nodos
        node_x, node_y, node_z = [], [], []
        node_text = []
        node_size = []
        node_colors = []
        
        # Calcular grados
        in_degrees = dict(subG_nx.in_degree())
        out_degrees = dict(subG_nx.out_degree())
        
        for node in subG_nx.nodes():
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            in_degree = in_degrees[node]
            out_degree = out_degrees[node]
            idx = id2idx[node]
            
            node_size.append(5 + 2*np.log1p(in_degree + out_degree))
            node_colors.append(in_degree)
            node_text.append(
                f'ID: {node}<br>'
                f'Índice: {idx}<br>'
                f'Conexiones entrantes: {in_degree}<br>'
                f'Conexiones salientes: {out_degree}'
            )
        
        # Crear traza de nodos
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title='Conexiones Entrantes'),
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Crear figura final
        fig = go.Figure(data=[*edge_traces, node_trace])
        
        # Configurar layout
        fig.update_layout(
            title=dict(
                text=f'Red Social - {subG_nx.number_of_nodes()} usuarios',
                x=0.5,
                y=0.95
            ),
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white'
        )
        
        # Guardar visualización
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                logging.info(f"Visualización guardada en {save_path}")
            except Exception as e:
                logging.error(f"Error guardando visualización: {e}")
        
        return fig
        
    except Exception as e:
        logging.error(f"Error durante la creación de la visualización: {str(e)}")
        return None

def visualize_graph_2d(g_igraph, id2idx, idx2id, max_nodes=1000, save_path=None):
    """
    Crea una visualización 2D del grafo con flechas direccionales.
    
    Args:
        g_igraph: Grafo de igraph
        id2idx: Diccionario de mapeo ID -> índice
        idx2id: Diccionario de mapeo índice -> ID
        max_nodes: Número máximo de nodos a visualizar
        save_path: Ruta donde guardar la imagen
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from tqdm import tqdm
    
    logging.info(f"Creando visualización 2D con máximo {max_nodes} nodos...")
    
    try:
        # Seleccionar nodos aleatoriamente
        available_ids = list(id2idx.keys())
        if len(available_ids) > max_nodes:
            selected_ids = random.sample(available_ids, max_nodes)
        else:
            selected_ids = available_ids
            
        # Crear subgrafo NetworkX
        G = nx.DiGraph()
        G.add_nodes_from(selected_ids)
        
        # Añadir aristas
        for orig_id in tqdm(selected_ids, desc="Procesando conexiones"):
            idx = id2idx[orig_id]
            neighbors = g_igraph.neighbors(idx, mode='out')
            for neighbor_idx in neighbors:
                if neighbor_idx in idx2id:
                    neighbor_id = idx2id[neighbor_idx]
                    if neighbor_id in G:
                        G.add_edge(orig_id, neighbor_id)
        
        logging.info(f"Subgrafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas")
        
        # Crear figura con subplots para manejar la colorbar
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Calcular layout
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)
        
        # Calcular grados para tamaños y colores
        node_degrees = [G.in_degree(node) for node in G.nodes()]
        node_sizes = [300 + 100 * np.log1p(deg) for deg in node_degrees]
        
        # Dibujar nodos
        nodes = nx.draw_networkx_nodes(G, pos, 
                                     node_size=node_sizes,
                                     node_color=node_degrees,
                                     cmap=plt.cm.viridis,
                                     alpha=0.7,
                                     ax=ax)
        
        # Dibujar aristas con flechas
        nx.draw_networkx_edges(G, pos,
                             edge_color='gray',
                             alpha=0.5,
                             arrows=True,
                             arrowsize=20,
                             width=0.5,
                             connectionstyle='arc3,rad=0.2',
                             ax=ax)
        
        # Añadir etiquetas para nodos con mayor grado
        labels = {}
        for node in G.nodes():
            if G.in_degree(node) > np.percentile(node_degrees, 75):
                labels[node] = str(node)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Configurar título y ejes
        ax.set_title(f'Red Social - {G.number_of_nodes()} usuarios\nFlechas indican "sigue a"')
        ax.set_axis_off()
        
        # Añadir colorbar correctamente
        plt.colorbar(nodes, ax=ax, label='Número de seguidores')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar o mostrar
        if save_path:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Visualización guardada en {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return G
        
    except Exception as e:
        logging.error(f"Error durante la creación de la visualización 2D: {str(e)}")
        return None

def detect_communities_girvan_newman(g_igraph, id2idx, idx2id, max_communities=10, save_path=None):
    """
    Detecta comunidades usando el algoritmo de comunidades multinivel.
    Convierte el grafo a no dirigido antes del análisis.
    
    Args:
        g_igraph: Grafo de igraph
        id2idx: Diccionario de mapeo ID -> índice
        idx2id: Diccionario de mapeo índice -> ID
        max_communities: Número máximo de comunidades a detectar
        save_path: Ruta para guardar la visualización
    """
    import igraph as ig
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    logging.info("Iniciando detección de comunidades...")
    
    try:
        # Crear subgrafo más pequeño para el análisis
        if g_igraph.vcount() > 1000:
            logging.info("Grafo muy grande, creando subgrafo para análisis...")
            selected_vertices = np.random.choice(g_igraph.vcount(), 1000, replace=False)
            subgraph = g_igraph.subgraph(selected_vertices)
        else:
            subgraph = g_igraph
        
        # Convertir a no dirigido
        logging.info("Convirtiendo grafo a no dirigido...")
        undirected_graph = subgraph.as_undirected(mode="collapse", combine_edges="sum")
            
        # Detectar comunidades
        logging.info("Detectando comunidades...")
        communities = undirected_graph.community_multilevel()
        
        # Obtener modularidad
        modularity = communities.modularity
        
        logging.info(f"Se encontraron {len(communities)} comunidades")
        logging.info(f"Modularidad: {modularity:.4f}")
        
        # Visualizar si se especifica una ruta
        if save_path:
            plt.figure(figsize=(15, 15))
            
            # Calcular layout
            layout = undirected_graph.layout_fruchterman_reingold(niter=100)
            
            # Colores para las comunidades
            colors = list(mcolors.TABLEAU_COLORS.values())
            
            # Dibujar nodos
            for idx, community in enumerate(communities):
                color = colors[idx % len(colors)]
                xs = [layout[vertex][0] for vertex in community]
                ys = [layout[vertex][1] for vertex in community]
                plt.scatter(xs, ys, c=color, label=f'Comunidad {idx+1} ({len(community)} nodos)', alpha=0.6)
            
            # Dibujar aristas
            edge_xs = []
            edge_ys = []
            for edge in undirected_graph.es:
                source, target = edge.tuple
                edge_xs.extend([layout[source][0], layout[target][0], None])
                edge_ys.extend([layout[source][1], layout[target][1], None])
            plt.plot(edge_xs, edge_ys, 'gray', alpha=0.2, linewidth=0.5)
            
            plt.title(f'Comunidades Detectadas\n{len(communities)} comunidades, Modularidad: {modularity:.4f}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.axis('off')
            
            # Guardar visualización
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Visualización guardada en {save_path}")
        
        # Convertir IDs de vértices a IDs originales
        communities_with_original_ids = []
        for community in communities:
            original_ids = [idx2id[subgraph.vs[vertex].index] for vertex in community]
            communities_with_original_ids.append(original_ids)
        
        return communities_with_original_ids, modularity
        
    except Exception as e:
        logging.error(f"Error durante la detección de comunidades: {str(e)}")
        return None, None
    
# --- Main Execution (Visualizer) ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Visualización de Grafo (Optimizado y Mejorado) ---")

    # 1. Cargar datos pre-procesados
    g_igraph_main, id2idx_main, idx2id_main, locations_main = load_processed_data(
        GRAPH_FILE, ID2IDX_FILE, IDX2ID_FILE, LOCATIONS_FILE
    )

    if g_igraph_main is None or id2idx_main is None or idx2id_main is None:
        logging.critical("No se pudieron cargar los datos esenciales del grafo. Terminando.")
        exit()
    
    # 2. Calcular Centralidades (hacerlo una vez)
    logging.info("\n--- Calculando Métricas Globales ---")
    pagerank_map = calculate_centralities(g_igraph_main, idx2id_main, measure='pagerank')
    # Descomentar con precaución: Betweenness es muy lento
    # betweenness_map = calculate_centralities(g_igraph_main, idx2id_main, measure='betweenness')
    eigenvector_map = calculate_centralities(g_igraph_main, idx2id_main, measure='eigenvector')

    # --- Ejecutar Visualizaciones ---
    
    # 3.1. Viz básica (subgrafo estático)
    logging.info("\n--- 3.1 Iniciando Visualización Básica ---")
    basic_graph_visualization(g_igraph_main, idx2id_main,
                              save_path="viz_basic_subgraph.png",
                              max_nodes=MAX_NODES_BASIC_VIZ)
    
    # 3.2. Comunidades (detección y viz estática)
    logging.info("\n--- 3.2 Iniciando Detección y Visualización de Comunidades ---")
    partition_main = community_visualization(g_igraph_main, id2idx_main, idx2id_main,
                                            save_path="viz_communities_subgraph.png",
                                            max_nodes=MAX_NODES_COMMUNITY_VIZ)
    
    # 3.3. Distribución de grados (global, log-log)
    logging.info("\n--- 3.3 Iniciando Visualización Distribución de Grados ---")
    degree_distribution_visualization(g_igraph_main,
                                      save_path="viz_degree_distribution.png")

    # 3.4. Viz interactiva (con PageRank)
    logging.info("\n--- 3.4 Iniciando Visualización Interactiva (PageRank) ---")
    interactive_visualization(g_igraph_main, id2idx_main, idx2id_main,
                              locations=locations_main,
                              communities=partition_main,
                              centrality_map=pagerank_map,
                              centrality_name='PageRank',
                              save_path="viz_interactive_pagerank.html",
                              max_nodes=MAX_NODES_INTERACTIVE_VIZ)

    # 3.5. Viz interactiva (con Eigenvector Centrality)
    logging.info("\n--- 3.5 Iniciando Visualización Interactiva (Eigenvector) ---")
    interactive_visualization(g_igraph_main, id2idx_main, idx2id_main,
                              locations=locations_main,
                              communities=partition_main,
                              centrality_map=eigenvector_map,
                              centrality_name='Eigenvector',
                              save_path="viz_interactive_eigenvector.html",
                              max_nodes=MAX_NODES_INTERACTIVE_VIZ)

    # 3.6. Viz geográfica (coloreada por PageRank)
    logging.info("\n--- 3.6 Iniciando Visualización Geográfica (Color: PageRank) ---")
    if locations_main:
        geo_visualization(g_igraph_main, id2idx_main, idx2id_main, locations_main,
                          centrality_map=pagerank_map, centrality_name='PageRank',
                          color_metric='centrality', # Decirle que coloree por centralidad
                          save_path="viz_geographic_pagerank.html",
                          max_nodes=MAX_NODES_GEO_VIZ)
    else:
        logging.warning("Saltando viz geo (PageRank): no se cargaron ubicaciones.")

    # 3.7. Viz geográfica
    logging.info("\n--- 3.7 Iniciando Visualización Geográfica (Color: Seguidores) ---")
    if locations_main:
        geo_visualization(g_igraph_main, id2idx_main, idx2id_main, locations_main,
                          centrality_map=pagerank_map, centrality_name='PageRank',
                          color_metric='followers', # Decirle que coloree por seguidores
                          save_path="viz_geographic_followers.html",
                          max_nodes=MAX_NODES_GEO_VIZ)
    else:
        logging.warning("Saltando viz geo (Seguidores): no se cargaron ubicaciones.")
    # Crear subgrafo NetworkX para las visualizaciones interactivas
    logging.info("\n--- Creando subgrafo NetworkX para visualizaciones interactivas ---")
    subG_nx, _ = create_sampled_networkx_subgraph(
        g_igraph_main, 
        idx2id_main, 
        MAX_NODES_INTERACTIVE_VIZ, 
        sampling_method='random'
    )
    
    fig = visualize_graph_relationships(
        g_igraph_main,
        id2idx_main,
        idx2id_main,
        max_nodes=1000,
        save_path="visualizations/graph_relationships.html"
    )
    
    # Ejemplo de uso
    G = visualize_graph_2d(
        g_igraph_main,
        id2idx_main,
        idx2id_main,
        max_nodes=1000,
        save_path="visualizations/network_2d.png"
    )
    
    # Ejemplo de uso
    communities, modularity = detect_communities_girvan_newman(
        g_igraph_main,
        id2idx_main,
        idx2id_main,
        max_communities=10,
        save_path="visualizations/communities_detected.png"
    )

    # Mostrar resultados
    if communities and modularity:
        print(f"\nNúmero de comunidades encontradas: {len(communities)}")
        print(f"Modularidad: {modularity:.4f}")
        
        for i, community in enumerate(communities, 1):
            print(f"Comunidad {i}: {len(community)} nodos")
    
    logging.info("\n--- Script de Visualización Completado ---")