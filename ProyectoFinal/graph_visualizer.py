import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
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
MST_PKL_FILE = os.path.join(PROCESS_DIR, "social_network_mst.igraph.pkl")
ID_MAP_PKL_FILE = os.path.join(PROCESS_DIR, "social_network_id_mappings.pkl")
LOG_FILE = os.path.join(OUTPUT_VIZ_DIR, "visualizer.log")

# Límites para las visualizaciones
MAX_NODES_STATIC_VIZ = 1500
MAX_NODES_GEO_VIZ = 5000
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

class GraphVisualizer:
    def __init__(self):
        self.g = None
        self.mst = None
        self.id2idx = None
        self.idx2id = None

    def load_data(self) -> bool:
        logging.info("--- Cargando Datos Pre-procesados ---")
        try:
            with open(GRAPH_PKL_FILE, 'rb') as f: self.g = pickle.load(f)
            logging.info(f"Grafo principal cargado: {self.g.vcount():,} nodos, {self.g.ecount():,} aristas.")
            with open(MST_PKL_FILE, 'rb') as f: self.mst = pickle.load(f)
            logging.info(f"Grafo MST cargado: {self.mst.vcount():,} nodos, {self.mst.ecount():,} aristas.")
            with open(ID_MAP_PKL_FILE, 'rb') as f:
                mappings = pickle.load(f)
                self.id2idx, self.idx2id = mappings['id2idx'], mappings['idx2id']
            logging.info(f"Mapeos de ID cargados ({len(self.id2idx):,} entradas).")
            if "community" not in self.g.vs.attributes(): logging.warning("Atributo 'community' no encontrado.")
            if "latitude" not in self.g.vs.attributes(): logging.warning("Atributo 'latitude' no encontrado.")
            return True
        except FileNotFoundError as e:
            logging.critical(f"Error fatal: Archivo no encontrado - {e.filename}. Asegúrate de que la ruta PROCESS_DIR ('{PROCESS_DIR}') sea correcta.")
            return False
        except Exception as e:
            logging.critical(f"Error fatal al cargar datos: {e}", exc_info=True)
            return False

    def _get_subgraph(self, max_nodes: int) -> ig.Graph:
        if self.g.vcount() <= max_nodes: return self.g
        logging.info(f"Grafo grande. Extrayendo subgrafo de {max_nodes} nodos del componente gigante...")
        giant = self.g.components(mode='weak').giant()
        if giant.vcount() <= max_nodes: return giant
        sampled_indices_in_giant = random.sample(range(giant.vcount()), max_nodes)
        original_indices = [giant.vs[i].index for i in sampled_indices_in_giant]
        return self.g.subgraph(original_indices)

    def visualize_subgraph(self, save_path="1_subgrafo_representativo.png", max_nodes=MAX_NODES_STATIC_VIZ):
        logging.info("Generando vista de subgrafo representativo...")
        subg = self._get_subgraph(max_nodes)
        layout = subg.layout_fruchterman_reingold()
        plot = ig.plot(subg, layout=layout, vertex_size=8, vertex_label=None, edge_width=0.5, bbox=(1200, 1200), margin=40)
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        plot.save(full_path)
        logging.info(f"Vista de subgrafo guardada en '{full_path}'.")

    def visualize_degree_histogram(self, save_path="2_histograma_grado.png"):
        logging.info("Generando histograma de distribución de grado...")
        in_degrees = self.g.degree(mode='in')
        out_degrees = self.g.degree(mode='out')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.hist(in_degrees, bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('Distribución de Grado de Entrada (Popularidad)')
        ax1.set_xlabel('Grado de Entrada (Nº de seguidores)')
        ax1.set_ylabel('Frecuencia (Nº de usuarios)')
        ax1.set_yscale('log')
        ax2.hist(out_degrees, bins=50, color='salmon', edgecolor='black')
        ax2.set_title('Distribución de Grado de Salida (Actividad)')
        ax2.set_xlabel('Grado de Salida (Nº de seguidos)')
        ax2.set_ylabel('Frecuencia (Nº de usuarios)')
        ax2.set_yscale('log')
        plt.tight_layout()
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        plt.savefig(full_path)
        logging.info(f"Histograma de grado guardado en '{full_path}'.")
        plt.close(fig)

    def visualize_geographic_heatmap(self, save_path="3_mapa_calor_geografico.html"):
        if 'latitude' not in self.g.vs.attributes():
            logging.warning("No hay datos de ubicación para el mapa de calor.")
            return
        logging.info("Generando mapa de calor de densidad geográfica...")
        valid_nodes = self.g.vs.select(latitude_ne=0.0, longitude_ne=0.0)
        coords = [[v['latitude'], v['longitude']] for v in valid_nodes]
        if coords:
            map_center = np.mean(coords, axis=0)
            m = folium.Map(location=map_center, zoom_start=2)
        else:
            m = folium.Map(location=[20, 0], zoom_start=2)
        HeatMap(coords, radius=10).add_to(m)
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        m.save(full_path)
        logging.info(f"Mapa de calor geográfico guardado en '{full_path}'.")

    # --- VISTA 4: Visualización del MST ---
    def visualize_mst(self, mode='geographic', save_path_prefix="4_viz_mst", max_nodes_geo=MAX_NODES_GEO_VIZ, max_nodes_struct=MAX_NODES_STATIC_VIZ):
        if not self.mst:
            logging.error("Grafo MST no cargado. No se puede generar la visualización.")
            return
        logging.info(f"Generando visualización de MST en modo '{mode}'...")
        if mode == 'structural':
            layout = self.mst.layout_fruchterman_reingold()
            fig = ig.plot(self.mst, layout=layout, vertex_size=5, vertex_label=None, edge_width=0.7, bbox=(1200, 1200), margin=20)
            full_path = os.path.join(OUTPUT_VIZ_DIR, f"{save_path_prefix}_estructural.png")
            fig.save(full_path)
            logging.info(f"Visualización de MST estructural guardada en '{full_path}'.")
        
        elif mode == 'geographic':
            logging.info("Filtrando nodos del MST con coordenadas geográficas válidas...")
            valid_geo_indices = [v.index for v in self.mst.vs if v['latitude'] != 0.0 and v['longitude'] != 0.0]
            logging.info(f"Encontrados {len(valid_geo_indices):,} nodos con datos geográficos en el MST.")

            if not valid_geo_indices:
                logging.warning("No hay nodos con datos geográficos en el MST. El mapa estará vacío.")
                return

            if len(valid_geo_indices) > max_nodes_geo:
                logging.info(f"Muestreando {max_nodes_geo} nodos para la visualización geográfica.")
                sampled_indices = random.sample(valid_geo_indices, max_nodes_geo)
            else:
                sampled_indices = valid_geo_indices
            
            target_mst = self.mst.subgraph(sampled_indices)
            logging.info(f"Subgrafo geográfico de MST creado con {target_mst.vcount()} nodos y {target_mst.ecount()} aristas.")

            edge_lons, edge_lats = [], []
            for edge in target_mst.es:
                source_v, target_v = target_mst.vs[edge.source], target_mst.vs[edge.target]
                if source_v["longitude"] != 0 and source_v["latitude"] != 0 and target_v["longitude"] != 0 and target_v["latitude"] != 0:
                    edge_lons.extend([source_v["longitude"], target_v["longitude"], None])
                    edge_lats.extend([source_v["latitude"], target_v["latitude"], None])
            
            if not edge_lons:
                logging.warning("No se generaron aristas para el mapa geográfico. Puede que los nodos muestreados no estén conectados entre sí.")

            fig = go.Figure(go.Scattermapbox(mode="lines", lon=edge_lons, lat=edge_lats, line=dict(width=1, color='blue')))
            fig.update_layout(title=f'Árbol de Expansión Mínima Geográfico ({target_mst.vcount()} nodos)', mapbox_style="carto-positron", margin={'r':0,'t':40,'l':0,'b':0})
            full_path = os.path.join(OUTPUT_VIZ_DIR, f"{save_path_prefix}_geografico.html")
            fig.write_html(full_path, include_plotlyjs='cdn')
            logging.info(f"Visualización de MST geográfico guardada en '{full_path}'.")

    def visualize_community_graph(self, save_path="5_grafo_por_comunidad.png", max_nodes=MAX_NODES_STATIC_VIZ):
        if 'community' not in self.g.vs.attributes():
            logging.warning("No se puede visualizar por comunidad, datos no disponibles.")
            return
        logging.info("Generando vista de grafo coloreado por comunidad con leyenda...")
        subg = self._get_subgraph(max_nodes)
        fig, ax = plt.subplots(figsize=(15, 12))
        vertex_colors, legend_patches = 'black', []
        if 'community' in subg.vs.attributes():
            communities = subg.vs['community']
            unique_communities = sorted(list(set(communities)))
            palette = plt.get_cmap('hsv')
            num_communities = len(unique_communities)
            if num_communities > 1:
                color_dict = {comm_id: palette(i / (num_communities - 1)) for i, comm_id in enumerate(unique_communities)}
            else:
                color_dict = {unique_communities[0]: palette(0.5)} if num_communities == 1 else {}
            vertex_colors = [color_dict.get(c, 'black') for c in communities]
            for comm_id, color in sorted(color_dict.items()):
                legend_patches.append(mpatches.Patch(color=color, label=f'Comunidad {comm_id}'))
        else:
            logging.warning("El subgrafo seleccionado no contiene el atributo 'community'.")
        layout = subg.layout_fruchterman_reingold()
        ig.plot(subg, layout=layout, target=ax, vertex_size=8, vertex_color=vertex_colors, vertex_label=None, edge_width=0.5)
        ax.set_title(f'Grafo de Comunidades ({subg.vcount()} nodos)', fontsize=16)
        if legend_patches:
            ax.legend(handles=legend_patches, title="Comunidades", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        plt.savefig(full_path)
        plt.close(fig)
        logging.info(f"Grafo por comunidad guardado en '{full_path}'.")
        
    def visualize_geographic_clusters(self, save_path="6_mapa_clusters_geograficos.html", max_nodes=MAX_NODES_GEO_VIZ):
        if 'latitude' not in self.g.vs.attributes() or 'community' not in self.g.vs.attributes():
            logging.warning("Faltan datos de ubicación o comunidad para el mapa de clusters.")
            return
        valid_nodes = self.g.vs.select(latitude_ne=0.0, longitude_ne=0.0)
        if len(valid_nodes) > max_nodes:
            valid_nodes = random.sample(list(valid_nodes), max_nodes)
        node_df = pd.DataFrame({
            'lat': [v['latitude'] for v in valid_nodes], 
            'lon': [v['longitude'] for v in valid_nodes],
            'community': [str(v['community']) for v in valid_nodes], 
            'id': [v['original_id'] for v in valid_nodes]
        })
        logging.info(f"Generando mapa interactivo de clusters con {len(node_df)} nodos...")
        fig = px.scatter_mapbox(
            node_df, lat="lat", lon="lon", color="community",
            hover_name="id", hover_data=["community"],
            mapbox_style="carto-positron",
            zoom=1, title=f"Clusters Geográficos de Usuarios ({len(node_df)} nodos)", 
            opacity=0.7, category_orders={"community": sorted(node_df['community'].unique())}
        )
        full_path = os.path.join(OUTPUT_VIZ_DIR, save_path)
        fig.write_html(full_path, include_plotlyjs='cdn')
        logging.info(f"Mapa de clusters geográficos guardado en '{full_path}'.")

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    logging.info("--- Iniciando Script de Visualización de Grafo ---")
    visualizer = GraphVisualizer()
    if visualizer.load_data():
        try:
            logging.info("\n--- Generando Visualizaciones Solicitadas ---")
            visualizer.visualize_subgraph()
            visualizer.visualize_degree_histogram()
            visualizer.visualize_geographic_heatmap()
            visualizer.visualize_mst(mode='structural')
            visualizer.visualize_mst(mode='geographic')
            visualizer.visualize_community_graph()
            visualizer.visualize_geographic_clusters()
            logging.info("\n--- Script de Visualización Completado ---")
            logging.info(f"Todas las visualizaciones han sido guardadas en el directorio: '{OUTPUT_VIZ_DIR}'")
        except Exception as e:
            logging.critical(f"Ocurrió un error durante la generación de una visualización: {e}", exc_info=True)
    else:
        logging.critical("No se pudo continuar debido a errores en la carga de datos.")