import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tqdm import tqdm
import community as community_louvain # pip install python-louvain
# import seaborn as sns # Can be used for degree dist plots if preferred
import time
import os
import random
from collections import Counter
import pickle # To load data saved by builder script
import logging

# --- Configuration ---
# Rutas a los archivos generados por graph_builder.py
PROCESS_DIR = './processed_data'
GRAPH_FILE = os.path.join(PROCESS_DIR, "social_network_graph_10M.gpickle")
LOCATIONS_FILE = os.path.join(PROCESS_DIR, "social_network_locations_10M.pkl")
OUTPUT_VIZ_DIR = './visualizations' # Directory to save visualizations
LOG_FILE = "graph_visualizer.log"

# Visualization Limits (Crucial for performance!)
MAX_NODES_BASIC_VIZ = 250       # Static Matplotlib network viz
MAX_NODES_COMMUNITY_VIZ = 1500  # Louvain + static viz
MAX_NODES_INTERACTIVE_VIZ = 2500 # Plotly interactive network
MAX_NODES_GEO_VIZ = 100_000     # Plotly map (can handle more points)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler()])

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

# --- Functions ---

def load_graph_data(graph_path, locations_path):
    """Loads the NetworkX graph and locations dictionary from saved files."""
    G = None
    locations = None

    # Load Graph
    logging.info(f"Cargando grafo NetworkX desde {graph_path}...")
    start_time = time.time()
    try:
        # Líneas corregidas:
        with open(graph_path, 'rb') as f: # Abrir en modo binario de lectura ('rb')
            G = pickle.load(f)
        logging.info(f"Grafo cargado en {time.time() - start_time:.2f} seg: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de grafo no encontrado: {graph_path}")
        return None, None # Cannot proceed without graph
    except Exception as e:
        logging.error(f"Error Crítico al cargar el grafo: {e}", exc_info=True)
        return None, None # Cannot proceed

    # Load Locations
    logging.info(f"Cargando ubicaciones desde {locations_path}...")
    start_time = time.time()
    try:
        with open(locations_path, 'rb') as f:
            locations = pickle.load(f)
        logging.info(f"Ubicaciones cargadas en {time.time() - start_time:.2f} seg: {len(locations)} localizaciones.")
    except FileNotFoundError:
        logging.warning(f"Advertencia: Archivo de ubicaciones no encontrado: {locations_path}. Las visualizaciones geográficas no funcionarán.")
        # Continue without locations
    except Exception as e:
        logging.error(f"Error al cargar las ubicaciones: {e}", exc_info=True)
        logging.warning("Continuando sin datos de ubicación debido a error de carga.")
        locations = None # Ensure locations is None if loading fails

    # Optional: Quick consistency check (sample check)
    if G and locations:
        sample_nodes = random.sample(list(G.nodes()), min(100, G.number_of_nodes()))
        missing_loc = [n for n in sample_nodes if n not in locations]
        if missing_loc:
             logging.warning(f"Advertencia: {len(missing_loc)} de 100 nodos de muestra no tienen ubicación en el archivo cargado (ej: {missing_loc[:5]}).")
        else:
             logging.info("Verificación rápida de consistencia de ubicaciones: OK (muestra).")


    return G, locations

def create_subgraph(G, max_nodes, sampling_method='random'):
    """Creates a subgraph by sampling nodes if G is too large."""
    if G.number_of_nodes() <= max_nodes:
        return G # No sampling needed

    logging.warning(f"Grafo original ({G.number_of_nodes()} nodos) excede el límite ({max_nodes}). Creando subgrafo...")
    nodes = list(G.nodes())
    if sampling_method == 'random':
        # Ensure k is not greater than the population size
        k = min(max_nodes, len(nodes))
        if k <= 0:
            logging.error("No hay nodos para muestrear.")
            return nx.DiGraph() # Return empty graph
        sampled_nodes = random.sample(nodes, k)
    # Add other sampling methods if needed (e.g., based on degree, PageRank)
    else:
        logging.error(f"Método de muestreo '{sampling_method}' no implementado. Usando 'random'.")
        k = min(max_nodes, len(nodes))
        if k <= 0: return nx.DiGraph()
        sampled_nodes = random.sample(nodes, k)

    subG = G.subgraph(sampled_nodes).copy() # Use copy for safety
    logging.info(f"Subgrafo creado con {subG.number_of_nodes()} nodos y {subG.number_of_edges()} aristas.")
    return subG


# --- Visualization Functions (Adapted from user's provided code with improvements) ---

def basic_graph_visualization(G, title="Subgrafo Red Social (Estático)", save_path=None, max_nodes=MAX_NODES_BASIC_VIZ):
    """Visualiza un subgrafo pequeño usando matplotlib."""
    if G is None:
        logging.error("Grafo no disponible para visualización básica.")
        return

    subG = create_subgraph(G, max_nodes)

    if subG.number_of_nodes() == 0:
        logging.warning("Subgrafo para visualización básica está vacío.")
        return

    plt.figure(figsize=(12, 10))
    logging.info(f"Calculando layout (spring) para {subG.number_of_nodes()} nodos...")
    try:
        # Adjust layout parameters for potentially dense graphs
        pos = nx.spring_layout(subG, seed=42, k=0.6/np.sqrt(subG.number_of_nodes()), iterations=30)
    except Exception as e:
        logging.error(f"Error calculando layout spring: {e}. Usando random layout.")
        pos = nx.random_layout(subG, seed=42)

    # Node color based on in-degree, size based on out-degree (log scaled)
    try:
        in_degree = dict(subG.in_degree())
        node_color = [in_degree.get(node, 0) for node in subG.nodes()]
        out_degree = dict(subG.out_degree())
        # Use log1p for scaling, add base size, ensure minimum size
        node_size = [15 + 10 * np.log1p(out_degree.get(node, 0)) for node in subG.nodes()]
    except Exception as e:
        logging.error(f"Error calculando grados para colorear/tamaño: {e}")
        node_color = 'skyblue' # Fallback
        node_size = 30       # Fallback

    logging.info("Dibujando grafo con Matplotlib...")
    nx.draw(
        subG,
        pos=pos,
        node_color=node_color,
        node_size=node_size,
        cmap=plt.cm.viridis,
        alpha=0.7,
        with_labels=False, # Labels usually unreadable for > 50 nodes
        arrows=True,
        arrowsize=8,
        edge_color='lightgray', # Lighter edges
        width=0.3 # Thinner edges
    )

    plt.title(f"{title}\n({subG.number_of_nodes()} nodos / {subG.number_of_edges()} aristas)", fontsize=14)
    plt.axis('off')

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=150, bbox_inches='tight') # Lower dpi for faster saving
            logging.info(f"Visualización básica guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización básica: {e}")
    else:
        plt.show()
    plt.close() # Close the figure to free memory

def community_visualization(G, save_path=None, max_nodes=MAX_NODES_COMMUNITY_VIZ):
    """Detecta y visualiza comunidades en un subgrafo usando Louvain."""
    if G is None:
        logging.error("Grafo no disponible para visualización de comunidades.")
        return None

    subG_orig = create_subgraph(G, max_nodes)

    if subG_orig.number_of_nodes() == 0:
        logging.warning("Subgrafo para detección de comunidades está vacío.")
        return None

    # Louvain works best on undirected graphs
    # Important: operate on a copy to avoid modifying the subgraph used elsewhere
    if subG_orig.is_directed():
        logging.info("Convirtiendo subgrafo a no dirigido para detección de comunidades Louvain.")
        # Use the undirected view for community detection, but keep original for layout?
        # Let's use undirected for both detection and layout for consistency here.
        subG_undirected = subG_orig.to_undirected()
    else:
        subG_undirected = subG_orig # Already undirected or graph view behaves as such

    # Remove isolates from the undirected version, as they don't form communities
    isolates = list(nx.isolates(subG_undirected))
    if isolates:
        logging.info(f"Removiendo {len(isolates)} nodos aislados del subgrafo no dirigido.")
        subG_undirected.remove_nodes_from(isolates)

    if subG_undirected.number_of_nodes() == 0:
        logging.warning("Subgrafo sin nodos no aislados. No se pueden detectar comunidades.")
        return None

    logging.info(f"Detectando comunidades (Louvain) en subgrafo de {subG_undirected.number_of_nodes()} nodos...")
    start_time = time.time()
    partition = {}
    modularity = None
    try:
        # Use random_state for reproducibility
        partition = community_louvain.best_partition(subG_undirected, random_state=42)
        elapsed_time = time.time() - start_time
        logging.info(f"Detección de comunidades completada en {elapsed_time:.2f} segundos")

        # Calculate Modularity
        if partition:
            modularity = community_louvain.modularity(partition, subG_undirected)
            logging.info(f"Modularidad de la partición: {modularity:.4f}")
        else:
            logging.warning("Partición de comunidad vacía.")

    except Exception as e:
        logging.error(f"Error durante la detección de comunidades Louvain: {e}", exc_info=True)
        return None # Cannot proceed without partition

    # Analyze communities
    community_counts = Counter(partition.values())
    num_communities = len(community_counts)
    logging.info(f"Número de comunidades detectadas: {num_communities}")

    if num_communities > 0:
      top_communities = community_counts.most_common(5)
      logging.info("Top 5 comunidades más grandes:")
      total_nodes_in_partition = sum(community_counts.values())
      for i, (community_id, count) in enumerate(top_communities):
          percentage = (count / total_nodes_in_partition * 100) if total_nodes_in_partition else 0
          logging.info(f"  {i+1}. Comunidad {community_id}: {count} nodos ({percentage:.2f}%)")
    else:
        logging.warning("No se detectaron comunidades.")
        # Still might visualize the graph colored by default if partition is empty

    # Visualization
    plt.figure(figsize=(14, 12))
    logging.info("Calculando layout para visualización de comunidades...")
    try:
        # Use the undirected graph for layout as well
        pos = nx.spring_layout(subG_undirected, seed=42, k=0.8/np.sqrt(subG_undirected.number_of_nodes()), iterations=40)
    except Exception as e:
        logging.error(f"Error calculando layout: {e}. Usando random layout.")
        pos = nx.random_layout(subG_undirected, seed=42)

    # Map community IDs to colors
    # Use a categorical colormap suitable for communities
    cmap = cm.get_cmap('tab20', num_communities if num_communities > 0 else 1)
    node_colors = [cmap(partition.get(node, -1)) for node in subG_undirected.nodes()] # Use get for safety

    logging.info("Dibujando nodos y aristas de comunidades...")
    nx.draw_networkx_nodes(
        subG_undirected, pos,
        node_color=node_colors,
        node_size=30, alpha=0.8
    )
    nx.draw_networkx_edges(
        subG_undirected, pos,
        edge_color='lightgray', width=0.2, alpha=0.5
    )

    mod_text = f"Modularidad: {modularity:.4f}" if modularity is not None else "Modularidad: N/A"
    plt.title(f"Comunidades en Subgrafo ({subG_undirected.number_of_nodes()} nodos) - Louvain\n"
              f"{num_communities} comunidades detectadas | {mod_text}", fontsize=14)
    plt.axis('off')

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Visualización de comunidades guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización de comunidades: {e}")
    else:
        plt.show()
    plt.close() # Close the figure

    # Return the partition corresponding to the NODES in the undirected subgraph
    return partition # Keys are node IDs, values are community IDs


def interactive_visualization(G, locations=None, communities=None, title="Red Social Interactiva (Subgrafo)", save_path=None, max_nodes=MAX_NODES_INTERACTIVE_VIZ):
    """Crea una visualización interactiva (Plotly) de un subgrafo."""
    if G is None:
        logging.error("Grafo no disponible para visualización interactiva.")
        return None

    subG = create_subgraph(G, max_nodes)

    if subG.number_of_nodes() == 0:
        logging.warning("Subgrafo para visualización interactiva está vacío.")
        return None

    logging.info(f"Preparando visualización interactiva para {subG.number_of_nodes()} nodos...")

    # Layout: Prioritize geographic if available and covers most nodes
    pos_3d = {}
    nodes_for_viz = list(subG.nodes()) # Start with all nodes in subgraph
    use_geo_layout = False

    if locations:
        nodes_with_loc = {n for n in subG.nodes() if n in locations}
        coverage = len(nodes_with_loc) / subG.number_of_nodes()
        logging.info(f"Cobertura de ubicaciones en subgrafo: {coverage*100:.1f}%")

        if coverage > 0.7: # Threshold for using geo layout
            logging.info("Usando coordenadas geográficas (proyección esférica) para layout 3D.")
            use_geo_layout = True
            nodes_for_viz = list(nodes_with_loc) # Only visualize nodes with locations
            if not nodes_for_viz:
                 logging.error("No hay nodos con ubicación en el subgrafo. No se puede usar layout geográfico.")
                 return None

            subG = subG.subgraph(nodes_for_viz).copy() # Filter subgraph further
            logging.info(f"Subgrafo filtrado a {subG.number_of_nodes()} nodos con ubicación para layout geo.")

            for node in tqdm(subG.nodes(), desc="Calculando coords 3D desde Lat/Lon"):
                lat, lon = locations[node]
                # Convert degrees to radians
                lat_rad, lon_rad = np.radians(lat), np.radians(lon)
                # Convert to Cartesian coordinates (unit sphere)
                x = np.cos(lat_rad) * np.cos(lon_rad)
                y = np.cos(lat_rad) * np.sin(lon_rad)
                z = np.sin(lat_rad)
                pos_3d[node] = (x, y, z)
        else:
            logging.info("Cobertura de ubicación insuficiente. Usando layout de red (spring_layout 3D).")

    if not use_geo_layout:
        logging.info("Calculando layout 3D (spring)...")
        try:
            # Ensure we use the nodes_for_viz list (which might just be subG.nodes if no geo)
            if subG.number_of_nodes() > 0:
                 pos_3d = nx.spring_layout(subG, seed=42, dim=3, k=0.5/np.sqrt(subG.number_of_nodes()), iterations=30)
            else:
                 logging.warning("Subgrafo vacío antes de calcular layout spring 3D.")
                 return None
        except Exception as e:
            logging.error(f"Error calculando layout 3D spring: {e}. Cancelando viz interactiva.")
            return None

    # Prepare edge data for Plotly
    edge_x, edge_y, edge_z = [], [], []
    logging.info("Preparando datos de aristas para Plotly...")
    for src, tgt in tqdm(subG.edges(), desc="Procesando aristas"):
        # Ensure both nodes exist in the calculated layout
        if src in pos_3d and tgt in pos_3d:
            x0, y0, z0 = pos_3d[src]
            x1, y1, z1 = pos_3d[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(180,180,180,0.3)', width=1),
        hoverinfo='none'
    )

    # Prepare node data
    node_x = [pos_3d[node][0] for node in subG.nodes()]
    node_y = [pos_3d[node][1] for node in subG.nodes()]
    node_z = [pos_3d[node][2] for node in subG.nodes()]

    node_texts = []
    node_colors = []
    color_title = 'Node Info'
    colorscale = 'Viridis'

    # Determine node color and hover text
    if communities:
        # Check if the provided communities map reasonably well to the current subgraph
        nodes_in_comm_subgraph = {n for n in subG.nodes() if n in communities}
        comm_coverage = len(nodes_in_comm_subgraph) / subG.number_of_nodes() if subG.number_of_nodes() > 0 else 0

        if comm_coverage > 0.7: # If most subgraph nodes have a community ID
            logging.info("Usando IDs de comunidad para colorear nodos.")
            # Default to -1 if node not in community partition (shouldn't happen if partition came from similar subgraph)
            node_colors = [communities.get(node, -1) for node in subG.nodes()]
            num_comms = len(set(node_colors) - {-1}) # Exclude -1 if present
            color_title = 'Comunidad ID'
            colorscale = 'Turbo' if num_comms > 10 else 'Viridis'
        else:
            logging.warning(f"Datos de comunidad ({comm_coverage*100:.1f}% cobertura) no coinciden bien con subgrafo interactivo. Usando grado de entrada.")
            communities = None # Force fallback

    if not communities: # Fallback to using in-degree for color
        logging.info("Usando grado de entrada (seguidores) para colorear nodos.")
        # Use original graph G for more accurate degree info if needed
        node_colors = [G.in_degree(node) for node in subG.nodes()]
        color_title = 'Grado Entrada (Seguidores)'
        colorscale = 'YlGnBu'

    # Generate hover text
    logging.info("Generando texto de hover...")
    for node in subG.nodes():
        in_deg = G.in_degree(node) # Degree from original graph
        out_deg = G.out_degree(node)
        hover_text = f'Usuario: {node}<br>Seguidores: {in_deg}<br>Siguiendo: {out_deg}'
        if communities and node in communities:
            hover_text += f'<br>Comunidad: {communities[node]}'
        if use_geo_layout and locations and node in locations:
             lat, lon = locations[node]
             hover_text += f'<br>Loc: ({lat:.3f}, {lon:.3f})'
        node_texts.append(hover_text)

    # Calculate node sizes (optional, use log scale)
    node_sizes = [5 + 3 * np.log1p(G.degree(node)) for node in subG.nodes()]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        name='Usuarios',
        marker=dict(
            symbol='circle',
            size=node_sizes, # Apply calculated sizes
            sizemode='diameter',
            color=node_colors,
            colorscale=colorscale,
            colorbar_title=color_title,
            line_width=0.5,
            line_color='rgb(50,50,50)'
        ),
        text=node_texts,
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'{title} ({subG.number_of_nodes()} nodos, {subG.number_of_edges()} aristas)',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, # Cleaner look
                               # Aspect ratio can be adjusted if needed
                               # aspectmode='cube'
                               ),
                    scene_camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)), # Adjust initial view
                    uirevision='constant' # Keep view state on updates
                    )
                )

    logging.info("Visualización interactiva preparada.")

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            fig.write_html(full_save_path)
            logging.info(f"Visualización interactiva guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización interactiva: {e}")
    # else: # Don't show by default, let main script decide or just save
        # fig.show()

    return fig


def degree_distribution_visualization(G, save_path=None):
    """Visualiza la distribución de grados (in y out) en escala log-log."""
    if G is None:
        logging.error("Grafo no disponible para analizar distribución de grados.")
        return
    if G.number_of_nodes() == 0:
        logging.warning("El grafo está vacío, no se puede calcular distribución de grados.")
        return

    logging.info("Calculando distribuciones de grado (esto puede tardar)...")
    start_time = time.time()

    try:
        # Efficiently get degrees as lists/iterators
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        if not in_degrees or not out_degrees:
             logging.warning("Grados cero o no calculables.")
             return

        # Use Counter for frequency distribution
        in_degree_counts = Counter(in_degrees)
        out_degree_counts = Counter(out_degrees)

        logging.info(f"Cálculo de grados completado en {time.time() - start_time:.2f} segundos.")

    except MemoryError:
        logging.error("Error de memoria al calcular grados. El grafo es demasiado grande para este análisis en memoria.")
        return
    except Exception as e:
        logging.error(f"Error calculando grados: {e}", exc_info=True)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot In-degree distribution
    if in_degree_counts:
        in_deg, in_cnt = zip(*sorted(in_degree_counts.items()))
        ax1.loglog(in_deg, in_cnt, 'o', markersize=3, alpha=0.6, color='blue')
        ax1.set_title('Distribución Grado de Entrada (Log-Log)')
        ax1.set_xlabel('Grado Entrada (k)')
        ax1.set_ylabel('Número de Nodos P(k)')
        ax1.grid(True, which="both", ls="--", linewidth=0.5)
    else:
        ax1.text(0.5, 0.5, 'No hay datos de grado de entrada', horizontalalignment='center', verticalalignment='center')
        ax1.set_title('Distribución Grado de Entrada (Log-Log)')


    # Plot Out-degree distribution
    if out_degree_counts:
        out_deg, out_cnt = zip(*sorted(out_degree_counts.items()))
        ax2.loglog(out_deg, out_cnt, 'o', markersize=3, alpha=0.6, color='red')
        ax2.set_title('Distribución Grado de Salida (Log-Log)')
        ax2.set_xlabel('Grado Salida (k)')
        ax2.set_ylabel('Número de Nodos P(k)')
        ax2.grid(True, which="both", ls="--", linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, 'No hay datos de grado de salida', horizontalalignment='center', verticalalignment='center')
        ax2.set_title('Distribución Grado de Salida (Log-Log)')


    plt.tight_layout()

    if save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
            logging.info(f"Visualización de distribución de grados guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización de grados: {e}")
    else:
        plt.show()
    plt.close() # Close the figure


def geo_visualization(G, locations, save_path=None, max_nodes=MAX_NODES_GEO_VIZ):
    """Visualiza la distribución geográfica de una muestra de usuarios."""
    if G is None:
        logging.error("Grafo no disponible para visualización geográfica.")
        return None
    if locations is None:
        logging.error("Ubicaciones no disponibles para visualización geográfica.")
        return None

    logging.info("Preparando datos para visualización geográfica...")

    # Filter nodes that are in the graph AND have a location
    nodes_with_loc_in_graph = {n for n in G.nodes() if n in locations}

    if not nodes_with_loc_in_graph:
        logging.warning("Ningún nodo en el grafo tiene ubicación válida. No se puede generar mapa.")
        return None

    # Sample if necessary
    if len(nodes_with_loc_in_graph) > max_nodes:
        logging.warning(f"Demasiados nodos con ubicación ({len(nodes_with_loc_in_graph)}). Mostrando muestra aleatoria de {max_nodes}.")
        sampled_nodes = random.sample(list(nodes_with_loc_in_graph), max_nodes)
    else:
        sampled_nodes = list(nodes_with_loc_in_graph)

    if not sampled_nodes:
        logging.warning("No quedan nodos después del muestreo geográfico.")
        return None

    # Create DataFrame for Plotly Express
    node_data = []
    logging.info(f"Extrayendo datos para {len(sampled_nodes)} nodos del mapa...")
    invalid_coords = 0
    for node in tqdm(sampled_nodes, desc="Procesando nodos para mapa"):
        try:
            lat, lon = locations[node]
            # Strict validation
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and -90 <= lat <= 90 and -180 <= lon <= 180:
                 in_deg = G.in_degree(node)
                 # Use log scale for size to prevent extremes, ensure minimum size
                 size = max(1, 5 * np.log1p(in_deg))
                 node_data.append({
                     'id': node,
                     'latitude': lat,
                     'longitude': lon,
                     'followers': in_deg,
                     # 'following': G.out_degree(node), # Add if needed in hover
                     'viz_size': size # Use a different name than plotly's 'size'
                 })
            else:
                invalid_coords += 1
                # logging.debug(f"Coordenadas inválidas o tipo incorrecto para nodo {node}: ({lat}, {lon})")
        except KeyError:
             logging.warning(f"Nodo {node} del grafo no encontrado en el diccionario de ubicaciones.")
        except Exception as e:
             logging.error(f"Error procesando datos geo para nodo {node}: {e}")
             invalid_coords += 1

    if invalid_coords > 0:
        logging.warning(f"Se ignoraron {invalid_coords} nodos debido a coordenadas inválidas o errores.")

    if not node_data:
        logging.error("No se pudieron extraer datos válidos para ningún nodo del mapa.")
        return None

    node_df = pd.DataFrame(node_data)
    logging.info(f"DataFrame creado con {len(node_df)} puntos válidos para el mapa.")

    # Create figure with Plotly Express
    logging.info("Generando mapa interactivo con Plotly Express...")
    fig = None
    try:
        fig = px.scatter_geo(node_df,
                            lat='latitude',
                            lon='longitude',
                            color='followers', # Color by influence (in-degree)
                            size='viz_size',    # Size by log(influence)
                            hover_name='id',
                            hover_data=['followers', 'latitude', 'longitude'],
                            projection='natural earth',
                            title=f'Distribución Geográfica (Muestra de {len(node_df)} usuarios)',
                            color_continuous_scale=px.colors.sequential.Plasma, # Example scale
                            size_max=15) # Control max marker size

        # Customize map appearance
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(
                showland=True, landcolor="rgb(229, 229, 229)",
                subunitcolor="rgb(255,255,255)",
                countrycolor="rgb(255,255,255)",
                showlakes=True, lakecolor='rgb(150, 190, 255)', # Lighter blue
                bgcolor='rgba(0,0,0,0)' # Transparent background for geo frame
            )
        )

        logging.info("Mapa geográfico generado.")

    except Exception as e:
        logging.error(f"Error al generar el mapa geográfico con Plotly Express: {e}", exc_info=True)
        return None # Return None if map generation fails

    if fig and save_path:
        try:
            full_save_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
            fig.write_html(full_save_path)
            logging.info(f"Visualización geográfica guardada en {full_save_path}")
        except Exception as e:
            logging.error(f"Error al guardar la visualización geográfica: {e}")
    # else: # Don't show automatically
        # if fig: fig.show()

    return fig


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Visualización de Grafo ---")

    # 1. Cargar datos pre-procesados
    G_complete, locations_data = load_graph_data(GRAPH_FILE, LOCATIONS_FILE)

    if G_complete is None:
        logging.critical("No se pudo cargar el grafo. Verifique el archivo .gpickle. Terminando script.")
        exit()

    # --- Ejecutar Visualizaciones / Análisis Visual ---

    # 2.1. Visualización básica estática (Subgrafo muy pequeño)
    logging.info("\n--- 2.1 Iniciando Visualización Básica (Subgrafo Pequeño) ---")
    basic_graph_visualization(G_complete,
                              save_path="viz_basic_subgraph.png",
                              max_nodes=MAX_NODES_BASIC_VIZ)

    # 2.2. Detección y visualización de comunidades (Subgrafo)
    logging.info("\n--- 2.2 Iniciando Detección y Visualización de Comunidades (Subgrafo) ---")
    # Run community detection and store the partition result for the analyzed subgraph
    partition_subgraph = community_visualization(G_complete,
                                                save_path="viz_communities_subgraph.png",
                                                max_nodes=MAX_NODES_COMMUNITY_VIZ)
    # Note: partition_subgraph only contains community info for the nodes in that specific subgraph

    # 2.3. Visualización de distribución de grados (Usa el grafo completo)
    logging.info("\n--- 2.3 Iniciando Visualización Distribución de Grados (Grafo Completo) ---")
    degree_distribution_visualization(G_complete,
                                      save_path="viz_degree_distribution.png")

    # 2.4. Visualización interactiva (Subgrafo)
    logging.info("\n--- 2.4 Iniciando Visualización Interactiva (Subgrafo) ---")
    # We can pass the partition_subgraph. The function will check if it's relevant.
    interactive_fig = interactive_visualization(G_complete,
                                                locations=locations_data,
                                                communities=partition_subgraph, # Pass results from 2.2
                                                save_path="viz_interactive_subgraph.html",
                                                max_nodes=MAX_NODES_INTERACTIVE_VIZ)
    # Optionally show the figure if running interactively and not just saving
    # if interactive_fig: interactive_fig.show()

    # 2.5. Visualización geográfica (Muestra de nodos)
    logging.info("\n--- 2.5 Iniciando Visualización Geográfica (Muestra) ---")
    if locations_data: # Only run if locations were loaded successfully
        geo_fig = geo_visualization(G_complete,
                                    locations_data,
                                    save_path="viz_geographic_sample.html",
                                    max_nodes=MAX_NODES_GEO_VIZ)
        # Optionally show the figure
        # if geo_fig: geo_fig.show()
    else:
        logging.warning("Saltando visualización geográfica porque no se cargaron las ubicaciones.")

    logging.info("\n--- Script de Visualización Completado ---")