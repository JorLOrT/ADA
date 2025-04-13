import os
import zipfile
import numpy as np
import time  # Para medir el tiempo de carga

# --- Configuración ---
NUM_USERS = 10_000_000
LOCATION_ZIP_FILE = 'dataset/10_million_location.txt.zip'
USER_ZIP_FILE = 'dataset/10_million_user.txt.zip'
LOCATION_TXT_FILE = '10_million_location.txt'
USER_TXT_FILE = '10_million_user.txt'

# --- Funciones Auxiliares ---

def unzip_file(zip_filepath, extract_to='.'):
    """Descomprime un archivo .zip si no existe el archivo descomprimido."""
    txt_filename = os.path.splitext(os.path.basename(zip_filepath))[0]
    txt_filepath = os.path.join(extract_to, txt_filename)

    if not os.path.exists(txt_filepath):
        print(f"Descomprimiendo {zip_filepath}...")
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Archivo descomprimido en: {txt_filepath}")
            return txt_filepath
        except FileNotFoundError:
            print(f"ERROR: Archivo ZIP no encontrado en {zip_filepath}")
            return None
        except Exception as e:
            print(f"ERROR: Ocurrió un error al descomprimir {zip_filepath}: {e}")
            return None
    else:
        print(f"Archivo {txt_filepath} ya existe. Saltando descompresión.")
        return txt_filepath

def load_data(location_filepath, user_filepath, num_users):
    """
    Carga los datos de ubicación y conexiones en estructuras eficientes.

    Args:
        location_filepath (str): Ruta al archivo de ubicaciones descomprimido.
        user_filepath (str): Ruta al archivo de usuarios (conexiones) descomprimido.
        num_users (int): Número total de usuarios.

    Returns:
        tuple: (locations, adjacency_list)
               locations (np.ndarray): Array de NumPy de shape (num_users, 2)
                                       con latitud y longitud. locations[i]
                                       corresponde al usuario i+1.
               adjacency_list (list): Lista de sets. adjacency_list[i] es un
                                      set de los IDs de usuarios que el
                                      usuario i+1 sigue.
               O None si ocurre un error.
    """
    if not location_filepath or not user_filepath:
        print("ERROR: Rutas de archivo no válidas.")
        return None

    print(f"\nIniciando carga de datos para {num_users} usuarios...")
    start_time = time.time()

    # --- Cargar Ubicaciones ---
    print(f"Cargando ubicaciones desde {location_filepath}...")
    # Pre-inicializar array de NumPy para eficiencia
    # Usar float32 puede ahorrar memoria si la precisión es suficiente
    locations = np.zeros((num_users, 2), dtype=np.float32)
    try:
        with open(location_filepath, 'r') as f_loc:
            for i, line in enumerate(f_loc):
                if i >= num_users:
                    print(f"ADVERTENCIA: El archivo de ubicaciones tiene más de {num_users} líneas. Ignorando extras.")
                    break
                try:
                    lat, lon = map(float, line.strip().split(','))
                    # El usuario i+1 (indexado desde 1) va en el índice i (indexado desde 0)
                    locations[i, 0] = lat
                    locations[i, 1] = lon
                except ValueError:
                    print(f"ADVERTENCIA: Formato inválido en línea {i+1} del archivo de ubicaciones: '{line.strip()}'. Saltando.")
                except IndexError:
                     print(f"ADVERTENCIA: Formato inválido (faltan datos) en línea {i+1} del archivo de ubicaciones: '{line.strip()}'. Saltando.")

                # Mostrar progreso
                if (i + 1) % 1_000_000 == 0:
                    print(f"  ... {i+1}/{num_users} ubicaciones cargadas.")

            # Verificar si se leyeron todas las ubicaciones esperadas
            if i + 1 < num_users:
                 print(f"ADVERTENCIA: Se esperaban {num_users} ubicaciones, pero solo se encontraron {i+1} en el archivo.")


    except FileNotFoundError:
        print(f"ERROR: Archivo de ubicaciones no encontrado en {location_filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Ocurrió un error inesperado al leer {location_filepath}: {e}")
        return None

    print(f"Ubicaciones cargadas en {time.time() - start_time:.2f} segundos.")
    loc_time = time.time()

    # --- Cargar Conexiones (Lista de Adyacencia) ---
    print(f"Cargando conexiones desde {user_filepath}...")
    # Pre-inicializar lista de sets vacíos
    adjacency_list = [set() for _ in range(num_users)]
    try:
        with open(user_filepath, 'r') as f_user:
            for i, line in enumerate(f_user):
                if i >= num_users:
                    print(f"ADVERTENCIA: El archivo de usuarios tiene más de {num_users} líneas. Ignorando extras.")
                    break

                line_stripped = line.strip()
                if not line_stripped: # Si la línea está vacía, el usuario no sigue a nadie
                    # adjacency_list[i] ya es un set vacío
                    continue

                try:
                    # Convertir la lista de seguidos a enteros y guardarlos en el set
                    # El usuario i+1 (indexado desde 1) va en el índice i (indexado desde 0)
                    # Los IDs en el archivo son 1-based, los almacenamos tal cual.
                    followed_users = set(map(int, line_stripped.split(',')))
                    adjacency_list[i] = followed_users
                except ValueError:
                    print(f"ADVERTENCIA: Formato inválido (no entero) en línea {i+1} del archivo de usuarios: '{line_stripped}'. Saltando línea.")

                # Mostrar progreso
                if (i + 1) % 1_000_000 == 0:
                    print(f"  ... {i+1}/{num_users} listas de adyacencia cargadas.")

             # Verificar si se leyeron todas las listas esperadas
            if i + 1 < num_users:
                 print(f"ADVERTENCIA: Se esperaban {num_users} listas de adyacencia, pero solo se encontraron {i+1} en el archivo.")


    except FileNotFoundError:
        print(f"ERROR: Archivo de usuarios no encontrado en {user_filepath}")
        return None
    except Exception as e:
        print(f"ERROR: Ocurrió un error inesperado al leer {user_filepath}: {e}")
        return None

    total_time = time.time() - start_time
    print(f"Conexiones cargadas en {time.time() - loc_time:.2f} segundos.")
    print(f"\nCarga de datos completada en {total_time:.2f} segundos.")

    return locations, adjacency_list

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("--- Iniciando Proceso de Carga de Datos de Red Social ---")

    # 1. Descomprimir archivos (si es necesario)
    location_txt = unzip_file(LOCATION_ZIP_FILE)
    user_txt = unzip_file(USER_ZIP_FILE)

    # 2. Cargar los datos
    loaded_data = None
    if location_txt and user_txt:
        loaded_data = load_data(location_txt, user_txt, NUM_USERS)

    # 3. Verificar y usar los datos (Ejemplo)
    if loaded_data:
        locations_data, adj_list_data = loaded_data
        print("\n--- Verificación de Datos Cargados ---")
        print(f"Tipo de datos de ubicaciones: {type(locations_data)}")
        print(f"Shape del array de ubicaciones: {locations_data.shape}")
        print(f"Tipo de datos de lista de adyacencia: {type(adj_list_data)}")
        print(f"Número de usuarios en lista de adyacencia: {len(adj_list_data)}")

        # Ejemplo de acceso:
        user_id_to_check = 5 # Verificar usuario con ID 5 (corresponde al índice 4)
        if user_id_to_check <= NUM_USERS and user_id_to_check > 0:
            user_index = user_id_to_check - 1
            print(f"\nEjemplo de acceso para Usuario ID {user_id_to_check}:")
            print(f"  Ubicación (Lat, Lon): {locations_data[user_index]}")
            print(f"  Usuarios seguidos por {user_id_to_check}: {adj_list_data[user_index]}") # Mostrar los primeros 10 si son muchos?
            # Ejemplo de chequeo de conexión: ¿El usuario 5 sigue al usuario 1?
            target_user_id = 1
            follows = target_user_id in adj_list_data[user_index]
            print(f"  ¿Usuario {user_id_to_check} sigue al usuario {target_user_id}? {follows}")
        else:
             print(f"\nID de usuario {user_id_to_check} fuera de rango (1-{NUM_USERS}).")

        print("\n¡Datos cargados exitosamente!")
    else:
        print("\nLa carga de datos falló.")

    print("\n--- Proceso Terminado ---")