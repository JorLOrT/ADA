import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import random
import pickle
import logging
import math

# --- Configuración Global ---
PROCESS_DIR = './processed_data_igraph'
OUTPUT_VIZ_DIR = './visualizations'

# Rutas de Archivos de ENTRADA
GRAPH_PKL_FILE = os.path.join(PROCESS_DIR, "social_network.igraph.pkl")
ID_MAP_PKL_FILE = os.path.join(PROCESS_DIR, "social_network_id_mappings.pkl")
LOG_FILE = os.path.join(OUTPUT_VIZ_DIR, "visualizer.log")

# Límites para las visualizaciones
MAX_NODES_INTERACTIVE_VIZ = 2000
MAX_NODES_GEO_VIZ = 2000
MAX_EDGES_GEO_VIZ = 1000

# --- Configuración del Logging ---
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def haversine(lon1, lat1, lon2, lat2):
    """Calcula la distancia de la gran esfera entre dos puntos en la tierra."""
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radio de la Tierra en km
    return c * r

class GraphVisualizer:
    """
    Clase para cargar un grafo pre-procesado y generar diversas visualizaciones.
    """
    def __init__(self):
        self.g = None
        self.id2idx = None
        self.pagerank = None

    def load_data(self):
        """Carga todos los datos procesados necesarios para la visualización."""
        logging.info("--- Cargando Datos Pre-procesados ---")
        try:
            with open(GRAPH_PKL_FILE, 'rb') as f:
                self.g = pickle.load(f)
            logging.info(f"Grafo cargado: {self.g.vcount()} nodos, {self.g.ecount()} aristas.")

            with open(ID_MAP_PKL_FILE, 'rb') as f:
                mappings = pickle.load(f)
                self.id2idx = mappings['id2idx']
            logging.info(f"Mapeos de ID cargados ({len(self.id2idx)} entradas).")

            logging.info("Pre-calculando PageRank para todos los nodos...")
            self.pagerank = self.g.pagerank()
            self.g.vs['pagerank'] = self.pagerank
            logging.info("PageRank calculado y añadido como atributo de nodo.")
            
            if "community" not in self.g.vs.attributes():
                logging.warning("Atributo 'community' no encontrado en el grafo.")
            if "latitude" not in self.g.vs.attributes():
                logging.warning("Atributo 'latitude' no encontrado en el grafo.")

            return True
        except FileNotFoundError as e:
            logging.critical(f"Error fatal: Archivo no encontrado - {e}. Asegúrate de que la ruta PROCESS_DIR sea correcta y hayas ejecutado el script de construcción.")
            return False
        except Exception as e:
            logging.critical(f"Error fatal al cargar datos: {e}", exc_info=True)
            return False

    def _get_subgraph(self, max_nodes):
        """Extrae un subgrafo para visualización, enfocado en el componente gigante."""
        if self.g.vcount() <= max_nodes:
            return self.g
        logging.info(f"Grafo grande. Extrayendo subgrafo de {max_nodes} nodos del componente gigante...")
        giant = self.g.components(mode='weak').giant()
        if giant.vcount() <= max_nodes: return giant
        sampled_indices_in_giant = random.sample(range(giant.vcount()), max_nodes)
        original_indices = [giant.vs[i].index for i in sampled_indices_in_giant]
        return self.g.subgraph(original_indices)

    # 1. Vista de Comunidades (Clustering)
    def visualize_communities_3d(self, save_path="1_viz_comunidades_3d.html", max_nodes=MAX_NODES_INTERACTIVE_VIZ):
        if not self.g or 'community' not in self.g.vs.attributes():
            logging.warning("No se pueden visualizar comunidades, datos no disponibles.")
            return

        subg = self._get_subgraph(max_nodes)
        logging.info(f"Preparando visualización 3D de comunidades para {subg.vcount()} nodos.")
        layout = subg.layout_fruchterman_reingold_3d(niter=100)
        
        edge_x, edge_y, edge_z = [], [], []
        for edge in subg.es:
            p0, p1 = layout[edge.source], layout[edge.target]
            edge_x.extend([p0[0], p1[0], None]); edge_y.extend([p0[1], p1[1], None]); edge_z.extend([p0[2], p1[2], None])

        edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='rgba(180,180,180,0.3)', width=1), hoverinfo='none')
        node_texts = [f"Usuario: {v['original_id']}<br>Comunidad: {v['community']}<br>PageRank: {v['pagerank']:.4e}" for v in subg.vs]
        node_trace = go.Scatter3d(
            x=[p[0] for p in layout], y=[p[1] for p in layout], z=[p[2] for p in layout],
            mode='markers', name='Usuarios',
            marker=dict(size=5, color=subg.vs['community'], colorscale='Viridis', colorbar=dict(title='ID Comunidad'), opacity=0.9),
            text=node_texts, hoverinfo='text'
        )
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f'Visualización de Comunidades ({subg.vcount()} nodos)', scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white')))
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.write_html(full_path, include_plotlyjs='cdn'); logging.info(f"Visualización de comunidades guardada en '{full_path}'.")

    # 2. Vista de Influencers
    def visualize_influencers_network(self, save_path="2_viz_influencers.png", top_n=20):
        if 'pagerank' not in self.g.vs.attributes():
            logging.error("PageRank no calculado. No se puede generar la vista de influencers.")
            return

        logging.info(f"Identificando los {top_n} influencers principales...")
        top_indices = sorted(range(self.g.vcount()), key=lambda i: self.g.vs[i]['pagerank'], reverse=True)[:top_n]
        neighbor_indices = set(top_indices)
        for idx in top_indices:
            neighbor_indices.update(self.g.neighbors(idx, mode='all'))
        
        subg = self.g.subgraph(list(neighbor_indices))
        logging.info(f"Visualizando red de influencers con {subg.vcount()} nodos y {subg.ecount()} aristas.")
        
        layout = subg.layout_fruchterman_reingold()
        top_v_indices_in_subg = subg.vs.select(lambda v: self.g.vs[v.index]['pagerank'] >= self.g.vs[top_indices[-1]]['pagerank']).indices
        node_sizes = [20 if v.index in top_v_indices_in_subg else 8 for v in subg.vs]
        node_colors = ['#FF4136' if v.index in top_v_indices_in_subg else '#0074D9' for v in subg.vs]

        fig = ig.plot(subg, layout=layout, vertex_size=node_sizes, vertex_color=node_colors, vertex_label=None, edge_width=0.5, bbox=(1000, 1000), margin=40)
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.save(full_path); logging.info(f"Vista de influencers guardada en '{full_path}'.")

    # 3. Vista Multinivel (Meta-Grafo de Comunidades)
    def visualize_community_meta_graph(self, save_path="3_viz_meta_grafo.png"):
        """Visualiza las comunidades como un meta-grafo para una vista de alto nivel."""
        if 'community' not in self.g.vs.attributes():
            logging.warning("No se pueden visualizar meta-grafo, datos de comunidad no disponibles.")
            return
        
        logging.info("Creando meta-grafo de comunidades...")
        
        meta_g = self.g.copy()
        meta_g.vs['node_count'] = 1
        
        # --- CORRECCIÓN APLICADA AQUÍ ---
        # Asignar un peso de 1 a cada arista ANTES de contraer/simplificar
        meta_g.es['weight'] = 1
        # --- FIN DE LA CORRECCIÓN ---

        communities = meta_g.vs['community']
        
        meta_g.contract_vertices(communities, combine_attrs=dict(
            pagerank="sum",
            node_count="sum"
        ))
        
        # Ahora simplify encontrará el atributo 'weight' y lo sumará correctamente.
        meta_g.simplify(combine_edges=sum)

        layout = meta_g.layout_fruchterman_reingold()
        
        # Esta línea ahora funcionará porque 'weight' existe.
        edge_widths = [math.log1p(w) for w in meta_g.es['weight']]
        node_sizes = [math.log1p(count) * 5 for count in meta_g.vs['node_count']]
        node_labels = [f"Com.{i}\n({count} nodos)" for i, count in enumerate(meta_g.vs['node_count'])]

        fig = ig.plot(meta_g, layout=layout,
                      vertex_size=node_sizes,
                      vertex_label=node_labels,
                      vertex_color='lightblue',
                      edge_width=edge_widths,
                      bbox=(1200, 1200), margin=60)
        
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.save(full_path); logging.info(f"Vista de meta-grafo guardada en '{full_path}'.")

    # 4. Vista de MST Estructural
    def visualize_structural_mst(self, save_path="4_viz_mst_estructural.png"):
        logging.info("Calculando y visualizando MST estructural del componente gigante...")
        giant = self.g.components(mode='weak').giant()
        mst = giant.spanning_tree(weights=None, return_tree=True)
        layout = mst.layout_fruchterman_reingold()
        fig = ig.plot(mst, layout=layout, vertex_size=5, vertex_label=None, edge_width=0.7, bbox=(1200, 1200), margin=20)
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.save(full_path); logging.info(f"Visualización de MST estructural guardada en '{full_path}'.")

    # 5. Vista Geográfica (Nodos)
    def visualize_geo_distribution(self, save_path="5_viz_geo_distribucion.html", max_nodes=MAX_NODES_GEO_VIZ):
        if 'latitude' not in self.g.vs.attributes():
            logging.warning("No hay datos de ubicación para la visualización geográfica.")
            return

        valid_nodes = self.g.vs.select(latitude_ne=0.0, longitude_ne=0.0)
        if len(valid_nodes) > max_nodes:
            valid_nodes = random.sample(list(valid_nodes), max_nodes)
        
        node_df = pd.DataFrame({
            'lat': [v['latitude'] for v in valid_nodes], 'lon': [v['longitude'] for v in valid_nodes],
            'community': [v['community'] for v in valid_nodes], 'id': [v['original_id'] for v in valid_nodes]
        })
        
        logging.info(f"Generando mapa interactivo de distribución con {len(node_df)} nodos...")
        fig = px.scatter_map(
            node_df, lat="lat", lon="lon", color="community",
            hover_name="id", hover_data=["community"], map_style="carto-positron",
            zoom=1, title=f"Distribución Geográfica de Usuarios ({len(node_df)} nodos)", opacity=0.6
        )
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.write_html(full_path, include_plotlyjs='cdn'); logging.info(f"Visualización geográfica guardada en '{full_path}'.")

    # 6. Vista de Conexiones Geográficas
    def visualize_geo_connections(self, save_path="6_viz_geo_conexiones.html", max_edges=MAX_EDGES_GEO_VIZ):
        if 'latitude' not in self.g.vs.attributes():
            logging.warning("No hay datos de ubicación para visualizar conexiones geográficas.")
            return

        logging.info(f"Preparando visualización de conexiones geográficas (máx {max_edges} aristas)...")
        valid_edges = self.g.es.select(lambda e: self.g.vs[e.source]['latitude'] != 0 and self.g.vs[e.target]['latitude'] != 0)
        sampled_edges = random.sample(list(valid_edges), min(len(valid_edges), max_edges))

        edge_lons, edge_lats = [], []
        for edge in sampled_edges:
            source_v, target_v = self.g.vs[edge.source], self.g.vs[edge.target]
            edge_lons.extend([source_v["longitude"], target_v["longitude"], None])
            edge_lats.extend([source_v["latitude"], target_v["latitude"], None])

        fig = go.Figure(go.Scattermap(mode="lines", lon=edge_lons, lat=edge_lats, line=dict(width=1, color='rgba(255,0,0,0.4)')))
        fig.update_layout(title=f'Conexiones Geográficas en la Red ({len(sampled_edges)} aristas)',
                          map_style="carto-darkmatter", margin={'r':0,'t':40,'l':0,'b':0})
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.write_html(full_path, include_plotlyjs='cdn'); logging.info(f"Visualización de conexiones geográficas guardada en '{full_path}'.")

    # 7. Vista de MST Geográfico
    def visualize_geographic_mst(self, save_path="7_viz_mst_geografico.html"):
        if 'latitude' not in self.g.vs.attributes():
            logging.error("Grafo o ubicaciones no disponibles para MST geográfico.")
            return

        logging.info("Calculando MST geográfico...")
        g_undirected = self.g.as_undirected()
        valid_edges = g_undirected.es.select(lambda e: g_undirected.vs[e.source]['latitude'] != 0 and g_undirected.vs[e.target]['latitude'] != 0)
        weights = [
            haversine(g_undirected.vs[edge.source]["longitude"], g_undirected.vs[edge.source]["latitude"],
                      g_undirected.vs[edge.target]["longitude"], g_undirected.vs[edge.target]["latitude"])
            for edge in valid_edges
        ]
        
        subg_for_mst = g_undirected.subgraph_edges(valid_edges, delete_vertices=False).copy()
        subg_for_mst.es['weight'] = weights
        
        giant_comp = subg_for_mst.components(mode='weak').giant()
        mst = giant_comp.spanning_tree(weights=giant_comp.es["weight"])

        edge_lons, edge_lats = [], []
        for edge in mst.es:
            source_v, target_v = mst.vs[edge.source], mst.vs[edge.target]
            edge_lons.extend([source_v["longitude"], target_v["longitude"], None])
            edge_lats.extend([source_v["latitude"], target_v["latitude"], None])

        fig = go.Figure(go.Scattermap(mode="lines", lon=edge_lons, lat=edge_lats, line=dict(width=1, color='blue')))
        fig.update_layout(title='Árbol de Expansión Mínima Geográfico', map_style="carto-positron", margin={'r':0,'t':40,'l':0,'b':0})
        
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.write_html(full_path, include_plotlyjs='cdn'); logging.info(f"Visualización de MST geográfico guardada en '{full_path}'.")


# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Visualización de Grafo ---")

    visualizer = GraphVisualizer()
    if visualizer.load_data():
        try:
            logging.info("\n--- Generando Visualizaciones ---")
            
            visualizer.visualize_communities_3d()
            visualizer.visualize_influencers_network()
            visualizer.visualize_community_meta_graph()
            visualizer.visualize_structural_mst()
            visualizer.visualize_geo_distribution()
            visualizer.visualize_geo_connections()
            visualizer.visualize_geographic_mst()
            
            logging.info("\n--- Script de Visualización Completado ---")
            logging.info(f"Todas las visualizaciones han sido guardadas en el directorio: '{OUTPUT_VIZ_DIR}'")

        except Exception as e:
            logging.critical(f"Ocurrió un error durante la generación de una visualización: {e}", exc_info=True)
    else:
        logging.critical("No se pudo continuar debido a errores en la carga de datos.")