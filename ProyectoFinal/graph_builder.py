import numpy as np
import igraph as ig
import networkx as nx # To convert for saving
import time
import os
import logging
import pickle

# --- Configuration ---
NUM_USERS = 10_000_000
LOCATION_TXT_FILE = 'D:/uSalle/Software/dataset/10_million_location.txt'
USER_TXT_FILE = 'D:/uSalle/Software/dataset/10_million_user.txt'
OUTPUT_DIR = './processed_data' # Directory to save output
GRAPH_NX_FILE = os.path.join(OUTPUT_DIR, "social_network_graph_10M.gpickle")
LOCATIONS_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_locations_10M.pkl")
LOG_FILE = "graph_builder.log"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler()])

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Functions ---

def load_locations_numpy(filepath, num_users):
    """
    Loads user locations efficiently using numpy.
    Assumes line 'i' corresponds to user ID 'i'.
    """
    logging.info(f"Iniciando carga de ubicaciones desde {filepath} para {num_users} usuarios.")
    start = time.time()
    locations = None
    try:
        # Pre-allocate numpy array for efficiency
        locations = np.zeros((num_users, 2), dtype=np.float32)
        count = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_users:
                    logging.warning(f"Se encontraron más líneas de las esperadas ({num_users}). Deteniendo en la línea {i}.")
                    break
                try:
                    lat, lon = map(float, line.strip().split(','))
                    # Basic validation
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        locations[i] = [lat, lon]
                        count += 1
                    else:
                        logging.warning(f"Coordenadas inválidas en línea {i+1}: {lat}, {lon}. Usando [0, 0].")
                        locations[i] = [0.0, 0.0] # Default or handle as needed

                except ValueError:
                    logging.error(f"Error al parsear la línea {i+1}: '{line.strip()}'. Usando [0, 0].")
                    locations[i] = [0.0, 0.0] # Default or handle as needed
                except Exception as e:
                     logging.error(f"Error inesperado procesando línea {i+1}: {e}")
                     locations[i] = [0.0, 0.0]

        elapsed = time.time() - start
        logging.info(f"Tiempo de carga de {count}/{num_users} ubicaciones válidas: {elapsed:.2f} segundos.")
        if count < num_users:
             logging.warning(f"Solo se cargaron {count} ubicaciones válidas de {num_users} esperadas.")
        return locations

    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de ubicaciones no encontrado en {filepath}.")
        return None
    except MemoryError:
        logging.error("Error Crítico: Memoria insuficiente para cargar las ubicaciones.")
        return None
    except Exception as e:
        logging.error(f"Error Crítico inesperado al cargar ubicaciones: {e}", exc_info=True)
        return None

def build_graph_igraph(user_filepath, num_users_hint):
    """
    Builds the graph using igraph for efficiency from user connections file.
    Maps original user IDs to sequential igraph vertex IDs.
    """
    logging.info(f"Iniciando construcción de grafo desde {user_filepath}.")
    start = time.time()
    user_ids = set()
    edges_raw = []
    lines_processed = 0

    try:
        with open(user_filepath, 'r') as f:
            # Using readline() might be slightly more memory efficient if lines are huge
            # but iteration is generally fine and pythonic.
            for i, line in enumerate(f):
                # Optional: Add a limit based on hint if file is unexpectedly huge
                # if i >= num_users_hint * 1.1: # Allow some margin
                #    logging.warning(f"Procesando más líneas ({i}) que el hint ({num_users_hint}).")
                #    # break # Optional: stop if file is way too big

                try:
                    ids = list(map(int, line.strip().split(',')))
                    if not ids: continue # Skip empty lines

                    src = ids[0]
                    user_ids.add(src)
                    # Add edges, ensuring src and dst are added to user_ids
                    for dst in ids[1:]:
                        user_ids.add(dst)
                        edges_raw.append((src, dst))
                    lines_processed += 1

                except ValueError:
                    logging.warning(f"Error al parsear IDs en línea {i+1}: '{line.strip()}'. Línea ignorada.")
                except Exception as e:
                    logging.error(f"Error inesperado procesando línea {i+1}: {e}. Línea ignorada.")

        logging.info(f"Procesadas {lines_processed} líneas del archivo de usuarios.")

        if not user_ids:
            logging.error("No se encontraron IDs de usuario válidos en el archivo.")
            return None, None, None

        # Create mappings between original IDs and sequential igraph indices (0-based)
        # Sorting ensures deterministic mapping
        sorted_user_ids = sorted(list(user_ids))
        id2idx = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
        idx2id = {idx: uid for uid, idx in id2idx.items()} # Inverse mapping

        logging.info(f"Mapeados {len(user_ids)} IDs únicos a índices.")

        # Convert raw edges using the index mapping
        edges = []
        invalid_edge_count = 0
        for src, dst in edges_raw:
            try:
                edges.append((id2idx[src], id2idx[dst]))
            except KeyError:
                # This shouldn't happen if logic above is correct, but good failsafe
                logging.warning(f"ID en edge ({src},{dst}) no encontrado en el mapeo. Edge ignorado.")
                invalid_edge_count += 1

        if invalid_edge_count > 0:
             logging.warning(f"Se ignoraron {invalid_edge_count} aristas con IDs no mapeados.")

        # Build the igraph Graph
        g = ig.Graph(n=len(user_ids), edges=edges, directed=True)

        elapsed = time.time() - start
        logging.info(f"Tiempo de creación del grafo igraph: {elapsed:.2f} segundos.")
        logging.info(f"Grafo igraph creado con {g.vcount()} nodos y {g.ecount()} aristas.")

        return g, id2idx, idx2id

    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de usuarios no encontrado en {user_filepath}.")
        return None, None, None
    except MemoryError:
        logging.error("Error Crítico: Memoria insuficiente para construir el grafo (listas intermedias).")
        return None, None, None
    except Exception as e:
        logging.error(f"Error Crítico inesperado al construir el grafo: {e}", exc_info=True)
        return None, None, None

def perform_basic_analysis(g, idx2id):
    """Performs basic graph analysis using igraph."""
    if g is None or not idx2id:
        logging.warning("Grafo no disponible para análisis básico.")
        return

    logging.info("Realizando análisis básico del grafo...")
    start = time.time()

    try:
        # Número de Nodos y Aristas (ya loggeado antes, pero bueno tenerlo aquí)
        num_nodes = g.vcount()
        num_edges = g.ecount()
        logging.info(f"Análisis - Nodos: {num_nodes}, Aristas: {num_edges}")

        if num_nodes == 0:
            logging.warning("Grafo vacío, análisis detenido.")
            return

        # Densidad del grafo
        # For directed graph, density = E / (V * (V - 1))
        density = g.density()
        logging.info(f"Análisis - Densidad del grafo: {density:.4e}") # Use scientific notation for small densities

        # IDs originales
        original_ids = list(idx2id.values())
        max_id = max(original_ids) if original_ids else "N/A"
        logging.info(f"Análisis - Usuario con el ID original más alto: {max_id}")

        # Grados (In-degree y Out-degree)
        # Using g.degree() is efficient in igraph
        logging.info("Calculando grados...")
        indegrees = g.degree(mode="in")
        outdegrees = g.degree(mode="out")

        # Usuario con más seguidores (mayor grado de entrada)
        if indegrees:
            idx_most_followed = np.argmax(indegrees)
            user_most_followed = idx2id[idx_most_followed]
            max_in_degree = indegrees[idx_most_followed]
            logging.info(f"Análisis - Usuario con más seguidores: ID {user_most_followed} ({max_in_degree} seguidores)")
        else:
             logging.warning("No se pudieron calcular los grados de entrada.")


        # Usuario que sigue a más personas (mayor grado de salida)
        if outdegrees:
            idx_follows_most = np.argmax(outdegrees)
            user_follows_most = idx2id[idx_follows_most]
            max_out_degree = outdegrees[idx_follows_most]
            logging.info(f"Análisis - Usuario que sigue a más personas: ID {user_follows_most} ({max_out_degree} seguidos)")
        else:
            logging.warning("No se pudieron calcular los grados de salida.")

        # Usuarios que no siguen a nadie (grado de salida 0)
        zero_outdegree_indices = [i for i, degree in enumerate(outdegrees) if degree == 0]
        users_no_follows = [idx2id[idx] for idx in zero_outdegree_indices]
        logging.info(f"Análisis - Usuarios que no siguen a nadie ({len(users_no_follows)}): {users_no_follows[:10]}...") # Mostrar solo los primeros

        # Usuarios que no son seguidos por nadie (grado de entrada 0)
        zero_indegree_indices = [i for i, degree in enumerate(indegrees) if degree == 0]
        users_no_followed = [idx2id[idx] for idx in zero_indegree_indices]
        logging.info(f"Análisis - Usuarios que no son seguidos por nadie ({len(users_no_followed)}): {users_no_followed[:10]}...") # Mostrar solo los primeros

        # Componentes Conectados (considerar débilmente conectado para grafos dirigidos)
        logging.info("Calculando componentes conectados...")
        components = g.components(mode='weak')
        num_components = len(components)
        logging.info(f"Análisis - Número de componentes conectados (débil): {num_components}")
        if num_components > 0:
            largest_component = components.giant()
            logging.info(f"Análisis - Tamaño del componente conectado más grande: {largest_component.vcount()} nodos ({largest_component.vcount()/num_nodes*100:.2f}%)")

        elapsed = time.time() - start
        logging.info(f"Tiempo de análisis básico: {elapsed:.2f} segundos.")

    except Exception as e:
        logging.error(f"Error durante el análisis básico del grafo: {e}", exc_info=True)


def convert_and_save_data(g_igraph, idx2id, locations_np, nx_filepath, loc_filepath):
    """
    Converts igraph graph to NetworkX, prepares location dictionary, and saves both.
    """
    if g_igraph is None:
        logging.error("Grafo igraph no disponible para convertir y guardar.")
        return False

    # 1. Convert igraph to NetworkX
    logging.info("Iniciando conversión de igraph a NetworkX...")
    start_conv = time.time()
    G_nx = None
    try:
        # Ensure node names in NetworkX are the *original* user IDs
        # Method 1: Create NX graph and add nodes/edges with original IDs
        G_nx = nx.DiGraph()
        logging.info("Añadiendo nodos a NetworkX con IDs originales...")
        # Add nodes first with original IDs
        for idx, original_id in idx2id.items():
             G_nx.add_node(original_id) # Add node with the original ID

        logging.info("Añadiendo aristas a NetworkX con IDs originales...")
        # Add edges using original IDs looked up from igraph indices
        for edge in g_igraph.es:
            src_idx, tgt_idx = edge.tuple
            try:
                src_orig_id = idx2id[src_idx]
                tgt_orig_id = idx2id[tgt_idx]
                G_nx.add_edge(src_orig_id, tgt_orig_id)
            except KeyError:
                 logging.warning(f"Índice de igraph no encontrado en idx2id durante conversión de arista: {src_idx} o {tgt_idx}. Arista ignorada.")


        # Method 2 (Simpler if igraph supports it directly with name attribute):
        # g_igraph.vs["name"] = [idx2id[i] for i in range(g_igraph.vcount())]
        # G_nx = g_igraph.to_networkx() # Check if this preserves names correctly

        elapsed_conv = time.time() - start_conv
        logging.info(f"Grafo convertido a NetworkX en {elapsed_conv:.2f} segundos.")
        logging.info(f"Grafo NetworkX: {G_nx.number_of_nodes()} nodos, {G_nx.number_of_edges()} aristas.")

    except Exception as e:
        logging.error(f"Error Crítico durante la conversión a NetworkX: {e}", exc_info=True)
        return False

    # 2. Save NetworkX graph
    logging.info(f"Guardando grafo NetworkX en {nx_filepath}...")
    start_save_nx = time.time()
    try:
        # Líneas corregidas:
        with open(nx_filepath, 'wb') as f:
            pickle.dump(G_nx, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed_save_nx = time.time() - start_save_nx
        logging.info(f"Grafo NetworkX guardado en {elapsed_save_nx:.2f} segundos.")
    except Exception as e:
        logging.error(f"Error Crítico al guardar el grafo NetworkX: {e}", exc_info=True)
        return False

    # 3. Prepare and save locations dictionary
    # Assuming locations_np[i] corresponds to the user with original ID 'i'
    if locations_np is not None:
        logging.info(f"Preparando y guardando diccionario de ubicaciones en {loc_filepath}...")
        start_save_loc = time.time()
        locations_dict = {}
        nodes_in_graph = set(G_nx.nodes()) # Use original IDs from NX graph
        valid_locations_count = 0
        try:
            # Iterate through the indices of the numpy array (assumed user IDs 0 to N-1)
            for user_id in range(len(locations_np)):
                # Only save locations for users actually present in the graph
                if user_id in nodes_in_graph:
                    # Convert numpy array row to tuple for pickling compatibility
                    locations_dict[user_id] = tuple(locations_np[user_id])
                    valid_locations_count += 1

            logging.info(f"Se prepararon {valid_locations_count} ubicaciones para usuarios presentes en el grafo.")

            with open(loc_filepath, 'wb') as f:
                pickle.dump(locations_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed_save_loc = time.time() - start_save_loc
            logging.info(f"Diccionario de ubicaciones guardado en {elapsed_save_loc:.2f} segundos.")
            return True
        except Exception as e:
            logging.error(f"Error al guardar las ubicaciones: {e}", exc_info=True)
            # Continue even if locations fail, but log it as critical data loss for geo viz
            return False # Indicate that part of the saving failed
    else:
        logging.warning("No hay datos de ubicación para guardar.")
        return True # Indicate success as there was nothing to fail on for locations


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Construcción de Grafo ---")

    # 1. Cargar Ubicaciones
    locations_data = load_locations_numpy(LOCATION_TXT_FILE, NUM_USERS)
    if locations_data is None:
         logging.warning("Continuando sin datos de ubicación debido a errores previos.")
         # Decide if execution should stop entirely if locations are critical
         # exit()

    # 2. Construir Grafo (igraph)
    graph_igraph, id_to_idx, idx_to_id = build_graph_igraph(USER_TXT_FILE, NUM_USERS)

    if graph_igraph is None:
        logging.critical("No se pudo construir el grafo igraph. Terminando.")
        exit()

    # 3. Análisis Básico (igraph)
    perform_basic_analysis(graph_igraph, idx_to_id)

    # 4. Convertir a NetworkX y Guardar Datos
    success = convert_and_save_data(graph_igraph, idx_to_id, locations_data,
                                    GRAPH_NX_FILE, LOCATIONS_PKL_FILE)

    if success:
        logging.info("--- Script de Construcción de Grafo Completado Exitosamente ---")
    else:
         logging.error("--- Script de Construcción de Grafo Completado con ERRORES al guardar datos ---")