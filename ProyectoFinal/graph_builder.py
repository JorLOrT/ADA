import numpy as np
import igraph as ig
import time
import os
import logging
import pickle

# --- Configuration ---
NUM_USERS = 10_000_000
# Ajusta estas rutas a tu ubicación real
LOCATION_TXT_FILE = './dataset/10_million_location.txt'
USER_TXT_FILE = './dataset/10_million_user.txt'
OUTPUT_DIR = './processed_data' # Directorio para guardar salida
# Cambiamos los nombres de archivo para reflejar el contenido
GRAPH_IGRAPH_FILE = os.path.join(OUTPUT_DIR, "social_network_graph_10M.igraph.pkl")
LOCATIONS_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_locations_10M.pkl")
IDX2ID_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_idx2id_10M.pkl")
ID2IDX_PKL_FILE = os.path.join(OUTPUT_DIR, "social_network_id2idx_10M.pkl") # Guardar ambos puede ser útil
LOG_FILE = "graph_builder.log"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), # mode='w' para sobrescribir log
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

def build_graph_igraph_memory_optimized(user_filepath):
    """
    Builds the graph using igraph efficiently, avoiding large intermediate edge lists.
    """
    logging.info(f"Iniciando construcción optimizada de grafo desde {user_filepath}.")
    start_mapping = time.time()
    user_ids = set()
    max_original_id_seen = -1

    # --- Primera pasada: Recolectar todos los IDs únicos ---
    logging.info("Primera pasada: Recolectando IDs únicos...")
    lines_processed_pass1 = 0
    try:
        with open(user_filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    ids_str = line.strip().split(',')
                    if not ids_str or not ids_str[0]: continue # Ignorar líneas vacías o mal formadas

                    # Convertir a int aquí para validación temprana
                    ids = [int(id_str) for id_str in ids_str]

                    src = ids[0]
                    user_ids.add(src)
                    max_original_id_seen = max(max_original_id_seen, src)
                    for dst in ids[1:]:
                        user_ids.add(dst)
                        max_original_id_seen = max(max_original_id_seen, dst)
                    lines_processed_pass1 += 1
                except ValueError:
                    logging.warning(f"Error al parsear IDs (ValueError) en línea {i+1}: '{line.strip()}'. Línea ignorada.")
                except Exception as e:
                    logging.error(f"Error inesperado procesando línea {i+1} en pasada 1: {e}. Línea ignorada.")

        logging.info(f"Primera pasada completada. {lines_processed_pass1} líneas procesadas.")

        if not user_ids:
            logging.error("No se encontraron IDs de usuario válidos en el archivo.")
            return None, None, None

        # --- Crear Mapeos ---
        sorted_user_ids = sorted(list(user_ids))
        id2idx = {uid: idx for idx, uid in enumerate(sorted_user_ids)}
        idx2id = {idx: uid for uid, idx in id2idx.items()} # Mapeo inverso
        num_unique_nodes = len(id2idx)
        logging.info(f"Mapeados {num_unique_nodes} IDs únicos a índices (0 a {num_unique_nodes-1}).")
        logging.info(f"ID original máximo encontrado: {max_original_id_seen}") # Útil para debug
        elapsed_mapping = time.time() - start_mapping
        logging.info(f"Tiempo para recolectar IDs y crear mapeos: {elapsed_mapping:.2f} segundos.")

    except FileNotFoundError:
        logging.error(f"Error Crítico: Archivo de usuarios no encontrado en {user_filepath}.")
        return None, None, None
    except MemoryError:
        logging.error("Error Crítico: Memoria insuficiente durante la recolección de IDs o creación de mapeos.")
        return None, None, None
    except Exception as e:
        logging.error(f"Error Crítico inesperado durante la pasada 1 o mapeo: {e}", exc_info=True)
        return None, None, None

    # --- Generador de Aristas (Segunda Pasada) ---
    def edge_generator(filepath, id_map):
        logging.info("Segunda pasada: Generando aristas (índices igraph)...")
        edge_count = 0
        lines_processed_pass2 = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    ids_str = line.strip().split(',')
                    if not ids_str or len(ids_str) < 2: continue # Necesita al menos src y un dst

                    # Convertir a int, manejar errores individuales
                    ids_int = []
                    valid_line = True
                    for id_s in ids_str:
                        try:
                            ids_int.append(int(id_s))
                        except ValueError:
                            logging.warning(f"ID inválido '{id_s}' en línea {i+1}. Ignorando ID.")
                            valid_line = False # O podríamos ignorar solo el ID
                            break # Ignorar toda la línea si un ID es malo? Decisión de diseño.

                    if not valid_line or len(ids_int) < 2: continue

                    src_orig = ids_int[0]
                    # Obtener índice igraph (manejar KeyError si un ID no estaba en la pasada 1, aunque no debería pasar)
                    src_idx = id_map.get(src_orig)
                    if src_idx is None:
                         logging.warning(f"ID origen {src_orig} de línea {i+1} no encontrado en mapeo. Ignorando aristas de esta línea.")
                         continue

                    for dst_orig in ids_int[1:]:
                        dst_idx = id_map.get(dst_orig)
                        if dst_idx is None:
                            logging.warning(f"ID destino {dst_orig} de línea {i+1} no encontrado en mapeo. Ignorando esta arista.")
                            continue

                        yield (src_idx, dst_idx)
                        edge_count += 1
                    lines_processed_pass2 +=1

                except Exception as e:
                    # Error inesperado leyendo la línea en la segunda pasada
                    logging.error(f"Error inesperado procesando línea {i+1} en pasada 2: {e}. Línea ignorada.")

        logging.info(f"Segunda pasada completada. {lines_processed_pass2} líneas procesadas.")
        logging.info(f"Generador entregó {edge_count} aristas.")

    # --- Construir Grafo igraph usando el Generador ---
    start_build = time.time()
    g = None
    try:
        logging.info("Construyendo grafo igraph desde el generador de aristas...")
        # Pasar el generador directamente a igraph
        g = ig.Graph(n=num_unique_nodes,
                     edges=edge_generator(user_filepath, id2idx),
                     directed=True)

        elapsed_build = time.time() - start_build
        logging.info(f"Tiempo de construcción del grafo igraph (desde generador): {elapsed_build:.2f} segundos.")
        logging.info(f"Grafo igraph final creado con {g.vcount()} nodos y {g.ecount()} aristas.")
        return g, id2idx, idx2id

    except MemoryError:
        # Todavía podría ocurrir si la estructura interna de igraph requiere mucha RAM
        logging.error("Error Crítico: Memoria insuficiente durante la construcción final del grafo igraph.")
        return None, None, None
    except Exception as e:
        logging.error(f"Error Crítico inesperado durante la construcción final del grafo igraph: {e}", exc_info=True)
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

def save_processed_data(g_igraph, id2idx, idx2id, locations_np,
                       graph_filepath, id2idx_filepath, idx2id_filepath, loc_filepath):
    """
    Saves the essential processed data: igraph graph, mappings, and locations.
    """
    success_flags = {'graph': False, 'id2idx': False, 'idx2id': False, 'locations': False}

    # 1. Save igraph graph
    if g_igraph is not None:
        logging.info(f"Guardando grafo igraph en {graph_filepath}...")
        start_save_g = time.time()
        try:
            # Usar el método nativo de igraph o pickle
            g_igraph.write_pickle(graph_filepath)
            # Alternativa:
            # with open(graph_filepath, 'wb') as f:
            #    pickle.dump(g_igraph, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed_save_g = time.time() - start_save_g
            logging.info(f"Grafo igraph guardado en {elapsed_save_g:.2f} segundos.")
            success_flags['graph'] = True
        except Exception as e:
            logging.error(f"Error Crítico al guardar el grafo igraph: {e}", exc_info=True)
    else:
        logging.error("Grafo igraph no disponible para guardar.")

    # 2. Save id2idx mapping
    if id2idx:
        logging.info(f"Guardando mapeo id->idx en {id2idx_filepath}...")
        start_save_map1 = time.time()
        try:
            with open(id2idx_filepath, 'wb') as f:
                pickle.dump(id2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed_save_map1 = time.time() - start_save_map1
            logging.info(f"Mapeo id->idx guardado en {elapsed_save_map1:.2f} segundos.")
            success_flags['id2idx'] = True
        except Exception as e:
            logging.error(f"Error al guardar el mapeo id->idx: {e}", exc_info=True)
    else:
         logging.warning("Mapeo id->idx no disponible para guardar.")

    # 3. Save idx2id mapping
    if idx2id:
        logging.info(f"Guardando mapeo idx->id en {idx2id_filepath}...")
        start_save_map2 = time.time()
        try:
            with open(idx2id_filepath, 'wb') as f:
                pickle.dump(idx2id, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed_save_map2 = time.time() - start_save_map2
            logging.info(f"Mapeo idx->id guardado en {elapsed_save_map2:.2f} segundos.")
            success_flags['idx2id'] = True
        except Exception as e:
            logging.error(f"Error al guardar el mapeo idx->id: {e}", exc_info=True)
    else:
        logging.warning("Mapeo idx->id no disponible para guardar.")


    # 4. Prepare and save locations dictionary (keyed by ORIGINAL ID)
    if locations_np is not None:
        logging.info(f"Preparando y guardando diccionario de ubicaciones en {loc_filepath}...")
        start_save_loc = time.time()
        locations_dict = {}
        valid_locations_count = 0
        # Asumimos que locations_np[i] es para el usuario con ID original i
        # Solo guardamos locaciones para los IDs que están en nuestro mapeo (y por ende en el grafo)
        nodes_in_graph_orig_ids = set(id2idx.keys()) # Obtener IDs originales del mapeo
        try:
            # Iterar hasta el máximo índice esperado O el tamaño del array numpy
            max_possible_id = len(locations_np)
            processed_ids = 0
            for user_id in range(max_possible_id):
                 # Optimización: Si ya procesamos todos los nodos del grafo, parar.
                 # if processed_ids >= len(nodes_in_graph_orig_ids): break

                 if user_id in nodes_in_graph_orig_ids:
                     loc_tuple = tuple(locations_np[user_id])
                     # Validar tupla de ubicación por si acaso
                     if (len(loc_tuple) == 2 and all(isinstance(x, (int, float, np.number)) for x in loc_tuple) and
                         -90 <= loc_tuple[0] <= 90 and -180 <= loc_tuple[1] <= 180):
                         locations_dict[user_id] = loc_tuple
                         valid_locations_count += 1
                     else:
                         logging.warning(f"Ubicación inválida encontrada para ID {user_id}: {loc_tuple}. Ignorada.")
                     processed_ids += 1


            logging.info(f"Se prepararon {valid_locations_count} ubicaciones para usuarios presentes en el grafo.")

            with open(loc_filepath, 'wb') as f:
                pickle.dump(locations_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed_save_loc = time.time() - start_save_loc
            logging.info(f"Diccionario de ubicaciones guardado en {elapsed_save_loc:.2f} segundos.")
            success_flags['locations'] = True
        except IndexError:
            logging.error(f"Error de índice al acceder a locations_np. ¿El tamaño ({len(locations_np)}) coincide con los IDs esperados?")
        except Exception as e:
            logging.error(f"Error al guardar las ubicaciones: {e}", exc_info=True)
    else:
        logging.warning("No hay datos de ubicación para guardar.")
        success_flags['locations'] = True # Considerar éxito si no había nada que guardar

    # Retornar True si al menos el grafo y los mapeos se guardaron
    return success_flags['graph'] and success_flags['id2idx'] and success_flags['idx2id']


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Construcción de Grafo (Optimizado para Memoria) ---")

    # 1. Cargar Ubicaciones
    locations_data = load_locations_numpy(LOCATION_TXT_FILE, NUM_USERS) # Se asume NUM_USERS es correcto
    # Si locations_data es None, se manejará en la función de guardado

    # 2. Construir Grafo (igraph) - Versión Optimizada
    # Ya no necesita NUM_USERS_HINT
    graph_igraph, id_to_idx, idx_to_id = build_graph_igraph_memory_optimized(USER_TXT_FILE)

    if graph_igraph is None:
        logging.critical("No se pudo construir el grafo igraph. Terminando.")
        exit()

    # 3. Análisis Básico (igraph) - Sin cambios, opera sobre igraph
    perform_basic_analysis(graph_igraph, idx_to_id)

    # 4. Guardar Datos Procesados (igraph, mapeos, ubicaciones)
    success = save_processed_data(graph_igraph, id_to_idx, idx_to_id, locations_data,
                                  GRAPH_IGRAPH_FILE, ID2IDX_PKL_FILE, IDX2ID_PKL_FILE, LOCATIONS_PKL_FILE)

    if success:
        logging.info("--- Script de Construcción de Grafo Completado Exitosamente ---")
    else:
         logging.error("--- Script de Construcción de Grafo Completado con ERRORES al guardar datos esenciales ---")