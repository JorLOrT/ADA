# graph_visualizer.py
# VERSIÓN FINAL Y VERIFICADA: Corrige el error en el histograma de grado.

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
from collections import Counter
from typing import Optional, Dict

# --- Límites para Visualizaciones ---
MAX_NODES_STATIC_VIZ = 1500
MAX_NODES_MST_VIZ = 500
MAX_NODES_GEO_CLUSTERS = 2500
MAX_NODES_HUBS_VIZ = 5000 

class GraphVisualizer:
    def __init__(self, process_dir='./processed_data', output_dir='./visualizations'):
        self.PROCESS_DIR = process_dir
        self.OUTPUT_VIZ_DIR = output_dir
        os.makedirs(self.OUTPUT_VIZ_DIR, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            file_handler = logging.FileHandler(os.path.join(self.OUTPUT_VIZ_DIR, "visualizer.log"), mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.g: Optional[ig.Graph] = None
        self.id2idx: Optional[Dict] = None
        self.idx2id: Optional[Dict] = None
        self._communities_calculated = False

    def load_data(self) -> bool:
        self.logger.info("--- Cargando Datos Procesados ---")
        try:
            with open(os.path.join(self.PROCESS_DIR, "social_network_graph_10M.igraph.pkl"), 'rb') as f: self.g = pickle.load(f)
            with open(os.path.join(self.PROCESS_DIR, "social_network_id2idx_10M.pkl"), 'rb') as f: self.id2idx = pickle.load(f)
            with open(os.path.join(self.PROCESS_DIR, "social_network_idx2id_10M.pkl"), 'rb') as f: self.idx2id = pickle.load(f)
            with open(os.path.join(self.PROCESS_DIR, "social_network_locations_10M.pkl"), 'rb') as f: locations = pickle.load(f)
            self.logger.info(f"Datos base cargados. Grafo con {self.g.vcount():,} nodos.")

            self.logger.info("Enriqueciendo grafo con atributos de vértice...")
            lats = [locations.get(self.idx2id.get(i), (0.0, 0.0))[0] for i in range(self.g.vcount())]
            lons = [locations.get(self.idx2id.get(i), (0.0, 0.0))[1] for i in range(self.g.vcount())]
            orig_ids = [self.idx2id.get(i, -1) for i in range(self.g.vcount())]
            
            self.g.vs['latitude'], self.g.vs['longitude'], self.g.vs['original_id'] = lats, lons, orig_ids
            self.logger.info("Atributos 'latitude', 'longitude', 'original_id' añadidos al grafo.")
            return True
        except FileNotFoundError as e:
            self.logger.critical(f"Error: Archivo no encontrado - {e.filename}. Ejecuta 'graph_builder.py' primero.")
            return False
        except Exception as e:
            self.logger.critical(f"Error fatal al cargar datos: {e}", exc_info=True)
            return False

    def _ensure_communities_calculated(self):
        if self._communities_calculated: return
        self.logger.info("Calculando comunidades (Leiden)...")
        g_undirected = self.g.as_undirected(combine_edges='first')
        weights = g_undirected.es['weight'] if 'weight' in g_undirected.edge_attributes() else None
        communities = g_undirected.community_leiden(weights=weights)
        self.g.vs['community'] = communities.membership
        self._communities_calculated = True
        self.logger.info(f"Comunidades calculadas y asignadas ({len(communities)} comunidades).")

    def _get_subgraph(self, max_nodes: int) -> ig.Graph:
        if self.g.vcount() <= max_nodes: return self.g
        giant = self.g.components(mode='weak').giant()
        nodes_to_sample_from = giant if giant.vcount() > 0 else self.g
        if nodes_to_sample_from.vcount() <= max_nodes: return nodes_to_sample_from
        sampled_vs_indices = random.sample(range(nodes_to_sample_from.vcount()), max_nodes)
        original_indices = [nodes_to_sample_from.vs[i].index for i in sampled_vs_indices]
        return self.g.subgraph(original_indices)
    
    def _get_subgraph_from_graph(self, graph, max_nodes):
        if graph.vcount() <= max_nodes: return graph
        sampled_indices = random.sample(range(graph.vcount()), max_nodes)
        return graph.subgraph(sampled_indices)
        
    # --- VISUALIZACIONES RESTAURADAS Y CORREGIDAS ---

    def visualize_subgraph(self, save_path="1_subgrafo_representativo.png"):
        self.logger.info("Generando vista de subgrafo representativo...")
        subg = self._get_subgraph(MAX_NODES_STATIC_VIZ)
        layout = subg.layout_fruchterman_reingold()
        plot = ig.plot(subg, layout=layout, vertex_size=8, vertex_label=None, edge_width=0.5, bbox=(1200, 1200), margin=40)
        plot.save(os.path.join(self.OUTPUT_VIZ_DIR, save_path))
        self.logger.info(f"Vista de subgrafo guardada en '{save_path}'.")

    def visualize_degree_loglog(self, save_path="2a_distribucion_grado_loglog.png"):
        self.logger.info("Generando distribución de grado Log-Log (Entrada vs. Salida)...")
        in_degrees, out_degrees = self.g.degree(mode='in'), self.g.degree(mode='out')
        in_counts, out_counts = Counter(d for d in in_degrees if d > 0), Counter(d for d in out_degrees if d > 0)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if in_counts:
            in_deg, in_cnt = zip(*sorted(in_counts.items()))
            ax.loglog(in_deg, in_cnt, 'o', markersize=5, alpha=0.6, label='Grado de Entrada (Seguidores)')
        if out_counts:
            out_deg, out_cnt = zip(*sorted(out_counts.items()))
            ax.loglog(out_deg, out_cnt, 'x', markersize=5, alpha=0.6, label='Grado de Salida (Seguidos)')
        ax.set_title('Distribución de Grado (Escala Log-Log)', fontsize=16)
        ax.set_xlabel('Grado (k)', fontsize=12); ax.set_ylabel('Número de Nodos con Grado k', fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.7); ax.legend()
        plt.savefig(os.path.join(self.OUTPUT_VIZ_DIR, save_path), dpi=200, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Gráfico log-log de grado guardado en '{save_path}'.")

    def visualize_degree_histogram(self, save_path="2b_histograma_grado.png"):
        self.logger.info("Generando histogramas de grado (Entrada vs. Salida)...")
        in_degrees = self.g.degree(mode='in')
        out_degrees = self.g.degree(mode='out')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        c
        ax1.hist(in_degrees, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
        ax1.set_title('Histograma de Grado de Entrada', fontsize=14)
        ax1.set_xlabel('Nº de Seguidores', fontsize=12)
        ax1.set_ylabel('Frecuencia', fontsize=12)
        ax1.set_yscale('log')
        
        # FIX: Usar la variable correcta 'out_degrees' para el segundo histograma.
        ax2.hist(out_degrees, bins=50, color='salmon', edgecolor='black', alpha=0.8)
        ax2.set_title('Histograma de Grado de Salida', fontsize=14)
        ax2.set_xlabel('Nº de Seguidos', fontsize=12)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_VIZ_DIR, save_path), dpi=200)
        plt.close(fig)
        self.logger.info(f"Histogramas de grado guardados en '{save_path}'.")

    def visualize_geographic_heatmap(self, save_path="3_mapa_calor_geografico.html"):
        self.logger.info("Generando mapa de calor geográfico de usuarios...")
        valid_nodes = self.g.vs.select(latitude_ne=0.0, longitude_ne=0.0)
        coords = [[v['latitude'], v['longitude']] for v in valid_nodes]
        if not coords: 
            self.logger.warning("No hay coordenadas válidas para el mapa de calor.")
            return
        m = folium.Map(location=np.mean(coords, axis=0), zoom_start=2)
        HeatMap(coords, radius=10).add_to(m)
        m.save(os.path.join(self.OUTPUT_VIZ_DIR, save_path))
        self.logger.info(f"Mapa de calor geográfico guardado en '{save_path}'.")

    def visualize_mst(self, save_path_prefix="4_viz_mst"):
        self.logger.info("Generando visualización de MST con mayor claridad...")
        if 'weight' not in self.g.edge_attributes(): 
            self.logger.error("Grafo no ponderado. No se puede calcular MST."); return
        
        mst_edge_ids = self.g.spanning_tree(weights='weight', return_tree=False)
        full_mst_g = self.g.subgraph_edges(mst_edge_ids, delete_vertices=False)
        
        mst_struct_g = self._get_subgraph_from_graph(full_mst_g, MAX_NODES_MST_VIZ)
        layout = mst_struct_g.layout_reingold_tilford()
        plot = ig.plot(mst_struct_g, layout=layout, vertex_size=6, vertex_label=None, edge_width=1.5, edge_color='#660099', bbox=(1200, 1200), margin=30)
        plot.save(os.path.join(self.OUTPUT_VIZ_DIR, f"{save_path_prefix}_estructural.png"))
        self.logger.info("Visualización de MST estructural guardada.")

        mst_geo_g = self._get_subgraph_from_graph(full_mst_g, MAX_NODES_MST_VIZ)
        edge_lons, edge_lats = [], []
        for edge in mst_geo_g.es:
            source_v, target_v = mst_geo_g.vs[edge.source], mst_geo_g.vs[edge.target]
            if source_v["latitude"] != 0 and target_v["latitude"] != 0:
                edge_lons.extend([source_v["longitude"], target_v["longitude"], None])
                edge_lats.extend([source_v["latitude"], target_v["latitude"], None])
        fig = go.Figure(go.Scattermap(mode="lines", lon=edge_lons, lat=edge_lats, line=dict(width=2, color='#FF00FF')))
        fig.update_layout(title=f'MST Geográfico ({mst_geo_g.vcount()} nodos)', map_style="carto-positron", margin={'r':0,'t':40,'l':0,'b':0})
        fig.write_html(os.path.join(self.OUTPUT_VIZ_DIR, f"{save_path_prefix}_geografico.html"), include_plotlyjs='cdn')
        self.logger.info("Visualización de MST geográfica guardada.")

    def visualize_community_graph(self, save_path="5_grafo_por_comunidad.png"):
        self._ensure_communities_calculated()
        self.logger.info("Generando grafo de comunidades con layout Kamada-Kawai...")
        subg = self._get_subgraph(MAX_NODES_STATIC_VIZ)
        layout = subg.layout_kamada_kawai()
        communities = subg.vs['community']
        unique_communities = sorted(list(set(communities)))
        palette = plt.get_cmap('hsv', len(unique_communities))
        color_dict = {comm_id: palette(i) for i, comm_id in enumerate(unique_communities)}
        fig, ax = plt.subplots(figsize=(15, 15))
        ig.plot(subg, layout=layout, target=ax, vertex_size=8, vertex_color=[color_dict.get(c, 'grey') for c in communities], vertex_label=None, edge_width=0.4, edge_color='#CCCCCC')
        ax.set_title(f'Grafo de Comunidades ({subg.vcount()} nodos)', fontsize=16)
        plt.savefig(os.path.join(self.OUTPUT_VIZ_DIR, save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Grafo por comunidad guardado en '{save_path}'.")

    def visualize_geographic_clusters(self, save_path="6_mapa_clusters_geograficos.html"):
        self._ensure_communities_calculated()
        self.logger.info("Generando mapa de clusters geográficos con menos nodos...")
        valid_nodes_indices = self.g.vs.select(latitude_ne=0.0, longitude_ne=0.0).indices
        if len(valid_nodes_indices) > MAX_NODES_GEO_CLUSTERS:
            valid_nodes_indices = random.sample(valid_nodes_indices, MAX_NODES_GEO_CLUSTERS)
        node_df = pd.DataFrame({
            'lat': [self.g.vs[i]['latitude'] for i in valid_nodes_indices],
            'lon': [self.g.vs[i]['longitude'] for i in valid_nodes_indices],
            'community': [str(self.g.vs[i]['community']) for i in valid_nodes_indices],
            'id': [self.g.vs[i]['original_id'] for i in valid_nodes_indices]
        })
        fig = px.scatter_map(node_df, lat="lat", lon="lon", color="community", hover_name="id", map_style="carto-positron", zoom=1, title=f"Clusters Geográficos ({len(node_df)} nodos de muestra)", opacity=0.7)
        fig.write_html(os.path.join(self.OUTPUT_VIZ_DIR, save_path), include_plotlyjs='cdn')
        self.logger.info(f"Mapa de clusters geográficos guardado en '{save_path}'.")
        
    def visualize_geographic_hubs(self, save_path="7_mapa_hubs_geograficos.html", top_n=100):
        self.logger.info(f"Generando mapa geográfico de los {top_n} hubs...")
        in_degrees, out_degrees = np.array(self.g.degree(mode='in')), np.array(self.g.degree(mode='out'))
        top_in, top_out = set(np.argsort(in_degrees)[-top_n:]), set(np.argsort(out_degrees)[-top_n:])
        hub_indices = list(top_in.union(top_out))
        hubs_with_geo = [self.g.vs[i] for i in hub_indices if self.g.vs[i]['latitude'] != 0.0]
        if not hubs_with_geo: 
            self.logger.warning("Ninguno de los hubs tiene datos de ubicación."); return
        hub_data = []
        for v in hubs_with_geo:
            is_pop, is_act = v.index in top_in, v.index in top_out
            hub_type, size = ('Popular y Activo', in_degrees[v.index] + out_degrees[v.index]) if is_pop and is_act else \
                             ('Popular', in_degrees[v.index]) if is_pop else ('Activo', out_degrees[v.index])
            hub_data.append({'lat': v['latitude'], 'lon': v['longitude'], 'id': v['original_id'],
                             'in_degree': in_degrees[v.index], 'out_degree': out_degrees[v.index],
                             'type': hub_type, 'size_metric': size})
        hub_df = pd.DataFrame(hub_data)
        fig = px.scatter_map(hub_df, lat="lat", lon="lon", color="type", size="size_metric", hover_name="id", 
                             hover_data={"in_degree": True, "out_degree": True, "size_metric": False},
                             map_style="carto-positron", zoom=1, title=f"Ubicación de los {top_n} Hubs",
                             opacity=0.8, color_discrete_map={'Popular': 'rgb(255,0,0)', 'Activo': 'rgb(0,0,255)', 'Popular y Activo': 'rgb(255,0,255)'})
        fig.update_layout(legend_title_text='Tipo de Hub')
        fig.write_html(os.path.join(self.OUTPUT_VIZ_DIR, save_path), include_plotlyjs='cdn')
        self.logger.info(f"Mapa de hubs geográficos guardado.")

    def visualize_community_density_histogram(self, save_path="8_histograma_densidad_comunidades.png"):
        self._ensure_communities_calculated()
        self.logger.info("Generando histograma de densidad de comunidades...")
        community_densities = []
        unique_comms, counts = np.unique(self.g.vs['community'], return_counts=True)
        top_communities_indices = np.argsort(counts)[-500:]
        top_communities = unique_comms[top_communities_indices]
        for comm_id in top_communities:
            if comm_id == -1: continue
            nodes_in_comm = self.g.vs.select(community_eq=comm_id)
            if len(nodes_in_comm) > 1:
                subg = self.g.subgraph(nodes_in_comm)
                community_densities.append(subg.density())
        if not community_densities: 
            self.logger.warning("No se pudo calcular la densidad de ninguna comunidad."); return
        plt.figure(figsize=(10, 6))
        plt.hist(community_densities, bins=50, color='purple', edgecolor='black', alpha=0.7)
        plt.title('Distribución de la Densidad de las Comunidades', fontsize=16); plt.xlabel('Densidad', fontsize=12)
        plt.ylabel('Frecuencia (Nº de Comunidades)', fontsize=12); plt.yscale('log')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.OUTPUT_VIZ_DIR, save_path), dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Histograma de densidad de comunidades guardado.")

    def visualize_active_users_by_community(self, save_path="9_usuarios_activos_por_comunidad.png", top_n_communities=15):
        self._ensure_communities_calculated()
        self.logger.info("Generando visualización de usuarios activos por comunidad...")
        out_degrees = np.array(self.g.degree(mode='out'))
        if len(out_degrees[out_degrees > 0]) == 0: 
            self.logger.warning("No hay usuarios con grado de salida > 0."); return
        activity_threshold = np.percentile(out_degrees[out_degrees > 0], 90)
        self.logger.info(f"Umbral de 'usuario activo' (P90 de grado de salida): > {activity_threshold:.0f} seguidos")
        active_user_indices = np.where(out_degrees > activity_threshold)[0]
        active_user_communities = [self.g.vs[i]['community'] for i in active_user_indices]
        community_activity = Counter(active_user_communities)
        if not community_activity: 
            self.logger.warning("No se encontraron usuarios activos en ninguna comunidad."); return
        most_active_communities = community_activity.most_common(top_n_communities)
        comm_ids, counts = zip(*most_active_communities)
        plt.figure(figsize=(12, 8))
        plt.barh([f'Comunidad {c}' for c in comm_ids], counts, color='teal', edgecolor='black')
        plt.xlabel('Número de Usuarios Activos (Top 10% por grado de salida)', fontsize=12); plt.ylabel('ID de Comunidad', fontsize=12)
        plt.title(f'Top {len(comm_ids)} Comunidades con Más Usuarios Activos', fontsize=16); plt.gca().invert_yaxis()
        if len(comm_ids) == 1:
            plt.figtext(0.5, 0.01, "Nota: La alta concentración en una comunidad es típica en redes sociales.", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(self.OUTPUT_VIZ_DIR, save_path), dpi=150)
        plt.close()
        self.logger.info(f"Gráfico de usuarios activos por comunidad guardado.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Main] %(message)s', handlers=[logging.StreamHandler()])
    logging.info("--- Iniciando Proceso de Visualización ---")
    visualizer = GraphVisualizer()
    if visualizer.load_data():
        logging.info("Datos cargados. Iniciando generación de TODAS las visualizaciones.")
        try:
            visualizer.visualize_subgraph()
            visualizer.visualize_degree_loglog()
            visualizer.visualize_degree_histogram()
            visualizer.visualize_geographic_heatmap()
            visualizer.visualize_mst()
            visualizer.visualize_community_graph()
            visualizer.visualize_geographic_clusters()
            visualizer.visualize_geographic_hubs()
            visualizer.visualize_community_density_histogram() 
            visualizer.visualize_active_users_by_community() 
            logging.info(f"\n--- Proceso de Visualización Completado ---")
            logging.info(f"Todas las visualizaciones han sido guardadas en: '{visualizer.OUTPUT_VIZ_DIR}'")
        except Exception as e:
            logging.critical(f"Ocurrió un error irrecuperable durante la visualización: {e}", exc_info=True)
    else:
        logging.critical("No se pudo continuar debido a errores en la carga de datos.")