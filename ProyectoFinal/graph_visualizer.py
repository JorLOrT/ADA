import os
import logging
import pickle
import time
import numpy as np
import igraph as ig
import networkx as nx
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import community_louvain as community

PROCESSED_DATA_DIR = './processed_data'
VISUALIZATIONS_DIR = './visualizations'

GRAPH_IGRAPH_FILE = os.path.join(PROCESSED_DATA_DIR, "social_network_graph.igraph.pkl")
IDX2ID_PKL_FILE = os.path.join(PROCESSED_DATA_DIR, "social_network_idx2id.pkl")
ID2IDX_PKL_FILE = os.path.join(PROCESSED_DATA_DIR, "social_network_id2idx.pkl")
LOCATIONS_PKL_FILE = os.path.join(PROCESSED_DATA_DIR, "social_network_locations.pkl")

COMMUNITY_PARTITION_FILE = os.path.join(PROCESSED_DATA_DIR, "community_partition.pkl")

LOG_FILE = os.path.join(VISUALIZATIONS_DIR, "graph_visualizer.log")

SAMPLE_SIZE_COMMUNITY_MAP = 200 
TOP_N_COMMUNITIES_MAP = 10 
TOP_N_COMMUNITIES_BAR = 20 

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) 
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [Visualizer] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

def load_base_data():
    """Carga los datos base necesarios (sin la partición de comunidades)."""
    logging.info("Iniciando la carga de datos base para visualización...")
    try:
        g = ig.Graph.Read_Pickle(GRAPH_IGRAPH_FILE)
        with open(IDX2ID_PKL_FILE, 'rb') as f: idx2id = pickle.load(f)
        with open(ID2IDX_PKL_FILE, 'rb') as f: id2idx = pickle.load(f)
        with open(LOCATIONS_PKL_FILE, 'rb') as f: locations = pickle.load(f)
        logging.info("Datos base cargados exitosamente.")
        return g, idx2id, id2idx, locations
    except FileNotFoundError as e:
        logging.error(f"Error: Archivo no encontrado - {e.filename}.")
        logging.error("Asegúrate de haber ejecutado 'graph_builder.py' primero.")
        return None, None, None, None

def detect_or_load_communities(g, partition_path):
    """
    Carga la partición de comunidades si el archivo existe. Si no, la detecta
    usando el algoritmo de Louvain, la guarda y luego la devuelve.
    """
    logging.info("--- Buscando o generando partición de comunidades ---")
    if os.path.exists(partition_path):
        logging.info(f"Archivo de partición encontrado. Cargando desde '{partition_path}'...")
        with open(partition_path, 'rb') as f:
            partition = pickle.load(f)
        logging.info("Partición de comunidades cargada exitosamente.")
        return partition

    logging.warning(f"Archivo de partición no encontrado en '{partition_path}'.")
    logging.info("Procediendo a detectar comunidades con el algoritmo de Louvain. Esto puede tardar...")
    start_total = time.time()
    g_undirected = g.as_undirected()
    logging.info("Convirtiendo grafo de igraph a NetworkX...")
    start_conv = time.time()
    nx_graph = nx.Graph(g_undirected.get_edgelist())
    logging.info(f"Conversión a NetworkX completada en {time.time() - start_conv:.2f}s.")
    logging.info("Ejecutando algoritmo de Louvain...")
    start_louvain = time.time()
    partition = community.best_partition(nx_graph, random_state=42)
    logging.info(f"Detección de comunidades completada en {time.time() - start_louvain:.2f}s.")
    if not partition:
        logging.error("La detección de comunidades no produjo resultados.")
        return None
    num_communities = len(set(partition.values()))
    modularity = community.modularity(partition, nx_graph)
    logging.info(f"  - Número de comunidades detectadas: {num_communities:,}")
    logging.info(f"  - Modularidad de la partición: {modularity:.4f}")
    logging.info(f"Guardando la partición de comunidades en '{partition_path}' para uso futuro.")
    with open(partition_path, 'wb') as f:
        pickle.dump(partition, f)
    logging.info(f"Proceso completo de detección y guardado de comunidades finalizado en {time.time() - start_total:.2f}s.")
    return partition

def create_heatmap(locations, output_path):
    """1. Genera un mapa de calor interactivo de la ubicación de los usuarios."""
    logging.info("--- 1. Creando mapa de calor de densidad de usuarios ---")
    location_points = [loc for loc in locations.values() if loc and (loc[0] != 0 or loc[1] != 0)]
    if not location_points:
        logging.warning("No se encontraron datos de ubicación válidos para generar el mapa de calor.")
        return
    map_center = [np.mean([p[0] for p in location_points]), np.mean([p[1] for p in location_points])]
    heatmap_map = folium.Map(location=map_center, zoom_start=4)
    HeatMap(location_points, radius=10).add_to(heatmap_map)
    heatmap_map.save(output_path)
    logging.info(f"Mapa de calor guardado en: {output_path}")

def plot_degree_distribution(g, output_path):
    """2. Genera un histograma de la distribución de grado de entrada y salida."""
    logging.info("--- 2. Creando histograma de distribución de grado ---")
    in_degrees = g.degree(mode='in')
    out_degrees = g.degree(mode='out')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("whitegrid")
    ax1.hist(in_degrees, bins=np.logspace(0, np.log10(max(in_degrees) + 1), 50), color='skyblue', edgecolor='black')
    ax1.set_title('Distribución de Grado de Entrada (Seguidores)', fontsize=14)
    ax1.set_xlabel('Número de Seguidores (Grado)', fontsize=12)
    ax1.set_ylabel('Número de Usuarios (Frecuencia)', fontsize=12)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax2.hist(out_degrees, bins=np.logspace(0, np.log10(max(out_degrees) + 1), 50), color='salmon', edgecolor='black')
    ax2.set_title('Distribución de Grado de Salida (Siguiendo)', fontsize=14)
    ax2.set_xlabel('Número de Usuarios Seguidos (Grado)', fontsize=12)
    ax2.set_ylabel('Número de Usuarios (Frecuencia)', fontsize=12)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    plt.suptitle('Análisis de Popularidad y Actividad de Usuarios', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Histograma de grado guardado en: {output_path}")

def create_community_map(partition, locations, idx2id, output_path):
    """3. Genera un mapa interactivo con capas por comunidad que se pueden activar/desactivar."""
    logging.info(f"--- 3. Creando mapa interactivo con capas por comunidad (Top {TOP_N_COMMUNITIES_MAP}) ---")

    communities = {}
    for node_idx, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node_idx)
    top_communities = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True)[:TOP_N_COMMUNITIES_MAP]
    
    palette = sns.color_palette('hsv', n_colors=len(top_communities)).as_hex()
    map_center = [40, -95] 
    community_map = folium.Map(location=map_center, zoom_start=4, tiles="cartodbpositron")

    logging.info(f"Creando una capa para cada una de las {len(top_communities)} comunidades más grandes...")

    for i, (comm_id, nodes) in enumerate(top_communities):
        color = palette[i]
        
        layer_name = f"Comunidad {comm_id} (Tamaño: {len(nodes):,})"
        feature_group = folium.FeatureGroup(name=layer_name)
        
        sampled_nodes_idx = np.random.choice(nodes, min(len(nodes), SAMPLE_SIZE_COMMUNITY_MAP), replace=False)
        
        for node_idx in sampled_nodes_idx:
            user_id = idx2id.get(node_idx)
            if user_id and user_id in locations:
                loc = locations[user_id]
                if loc and (loc[0] != 0 or loc[1] != 0):
                    folium.CircleMarker(
                        location=loc,
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"<b>Usuario ID:</b> {user_id}<br><b>Comunidad:</b> {comm_id}"
                    ).add_to(feature_group)
        
        feature_group.add_to(community_map)
                    
    folium.LayerControl().add_to(community_map)
    
    community_map.save(output_path)
    logging.info(f"Mapa interactivo de comunidades con capas guardado en: {output_path}")

def plot_community_sizes(partition, output_path):
    """4. Genera un gráfico de barras mostrando el tamaño de las comunidades más grandes."""
    logging.info(f"--- 4. Creando gráfico de barras de tamaño de comunidades (Top {TOP_N_COMMUNITIES_BAR}) ---")
    counts = Counter(partition.values())
    top_communities = counts.most_common(TOP_N_COMMUNITIES_BAR)
    if not top_communities:
        logging.warning("No hay datos de comunidad para graficar.")
        return
    comm_ids, sizes = zip(*top_communities)
    
    str_comm_ids = [f"C-{cid}" for cid in comm_ids]
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x=str_comm_ids, y=list(sizes), hue=str_comm_ids, palette="viridis", legend=False)
    
    ax.set_title(f'Tamaño de las {TOP_N_COMMUNITIES_BAR} Comunidades Más Grandes', fontsize=18, weight='bold')
    ax.set_xlabel('ID de Comunidad', fontsize=14)
    ax.set_ylabel('Número de Usuarios', fontsize=14)
    
    plt.xticks(rotation=45, ha="right")
    
    for i, size in enumerate(sizes):
        ax.text(i, size + (max(sizes) * 0.01), f'{size:,}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico de barras de comunidades guardado en: {output_path}")

if __name__ == "__main__":
    logging.info("--- INICIANDO SCRIPT DE VISUALIZACIÓN DE GRAFO ---")
    g, idx2id, id2idx, locations = load_base_data()
    if all(data is not None for data in [g, idx2id, id2idx, locations]):
        create_heatmap(locations, os.path.join(VISUALIZATIONS_DIR, '1_user_heatmap.html'))
        plot_degree_distribution(g, os.path.join(VISUALIZATIONS_DIR, '2_degree_distribution.png'))
        partition = detect_or_load_communities(g, COMMUNITY_PARTITION_FILE)
        if partition:
            create_community_map(partition, locations, idx2id, os.path.join(VISUALIZATIONS_DIR, '3_community.html'))
        else:
            logging.error("No se pudieron generar las comunidades.")

        logging.info("--- SCRIPT DE VISUALIZACIÓN FINALIZADO ---")
        logging.info(f"Todos los archivos han sido guardados en el directorio: '{VISUALIZATIONS_DIR}'")
    else:
        logging.critical("No se pudieron cargar los datos necesarios. Abortando script de visualización.")
