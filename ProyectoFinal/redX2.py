import numpy as np
import igraph as ig
import time
import networkx as nx
import matplotlib.pyplot as plt

NUM_USERS = 10_000_000
LOCATION_TXT_FILE = './dataset/10_million_location.txt'
USER_TXT_FILE = './dataset/10_million_user.txt'

def load_locations_numpy(filepath, num_users):
    start = time.time()
    locations = np.zeros((num_users, 2), dtype=np.float32)
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_users:
                break
            lat, lon = map(float, line.strip().split(','))
            locations[i] = [lat, lon]
    elapsed = time.time() - start
    print(f"Tiempo de carga de ubicaciones: {elapsed:.2f} segundos")
    return locations

def build_graph_igraph(user_filepath, num_users):
    start = time.time()
    user_ids = set()
    edges_raw = []
    with open(user_filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_users:
                break
            ids = list(map(int, line.strip().split(',')))
            src = ids[0]
            user_ids.add(src)
            for dst in ids[1:]:
                user_ids.add(dst)
                edges_raw.append((src, dst))
    id2idx = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
    idx2id = {idx: uid for uid, idx in id2idx.items()}
    edges = [(id2idx[src], id2idx[dst]) for src, dst in edges_raw]
    g = ig.Graph(directed=True)
    g.add_vertices(len(user_ids))
    g.add_edges(edges)
    elapsed = time.time() - start
    print(f"Tiempo de creación del grafo: {elapsed:.2f} segundos")
    return g, id2idx, idx2id

if __name__ == "__main__":
    print("Cargando ubicaciones...")
    locations = load_locations_numpy(LOCATION_TXT_FILE, NUM_USERS)
    print("Cargando conexiones y construyendo grafo...")
    g, id2idx, idx2id = build_graph_igraph(USER_TXT_FILE, NUM_USERS)
    print(f"Grafo creado con {g.vcount()} nodos y {g.ecount()} aristas.")

    # Usuario con el ID más alto
    max_id = max(id2idx.keys())
    print(f"Usuario con el ID más alto: {max_id}")

    # Usuario con más seguidores (mayor grado de entrada)
    indegrees = g.degree(mode="IN")
    idx_most_followed = np.argmax(indegrees)
    user_most_followed = idx2id[idx_most_followed]
    print(f"Usuario con más seguidores: {user_most_followed} ({indegrees[idx_most_followed]} seguidores)")

    # Usuario que sigue a más personas (mayor grado de salida)
    outdegrees = g.degree(mode="OUT")
    idx_follows_most = np.argmax(outdegrees)
    user_follows_most = idx2id[idx_follows_most]
    print(f"Usuario que sigue a más personas: {user_follows_most} ({outdegrees[idx_follows_most]} seguidos)")

    # Usuarios que no siguen a nadie (grado de salida 0)
    no_follows_idx = np.where(np.array(outdegrees) == 0)[0]
    users_no_follows = [idx2id[idx] for idx in no_follows_idx]
    print(f"Usuarios que no siguen a nadie: {users_no_follows[:10]}... (total: {len(users_no_follows)})")

    # Usuarios que no son seguidos por nadie (grado de entrada 0)
    no_followed_idx = np.where(np.array(indegrees) == 0)[0]
    users_no_followed = [idx2id[idx] for idx in no_followed_idx]
    print(f"Usuarios que no son seguidos por nadie: {users_no_followed[:10]}... (total: {len(users_no_followed)})")

    print("Detectando comunidades en un subgrafo pequeño...")
    sub_nodes = range(min(50, g.vcount()))
    subg = g.subgraph(sub_nodes)
    layout = [(locations[idx2id[n]][1], locations[idx2id[n]][0]) for n in sub_nodes]

    # Detección de comunidades
    dendrogram = subg.community_edge_betweenness()
    clusters = dendrogram.as_clustering()
    membership = clusters.membership

    # Visualización de comunidades
    color_palette = ig.drawing.colors.ClusterColoringPalette(len(clusters))
    visual_style = {
        "vertex_size": 10,
        "vertex_color": [color_palette.get(membership[i]) for i in range(len(subg.vs))],
        "layout": layout,
        "bbox": (600, 600),
        "margin": 40,
        "vertex_label": list(sub_nodes)
    }
    ig.plot(subg, **visual_style, target="comunidades.png")
    print("Comunidades detectadas y graficadas en 'comunidades.png'.")

    # Visualización con matplotlib
    print("Mostrando subgrafo con matplotlib...")
    nx_subg = subg.to_networkx()
    pos = {n: layout[i] for i, n in enumerate(sub_nodes)}
    plt.figure(figsize=(8, 8))
    nx.draw(nx_subg, pos, node_size=10, arrowsize=5, with_labels=True, font_size=8,
            node_color=[color_palette.get(membership[i]) for i in range(len(subg.vs))],
            edge_color='gray')
    plt.title("Comunidades en subgrafo de usuarios (matplotlib)")
    plt.show()