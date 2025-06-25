# -*- coding: utf-8 -*-
"""
Este módulo implementa la detección de comunidades.
"""
from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

# coding=utf-8

class Status(object):
    """
    Para manejar varios datos en una sola estructura.

    Podría ser reemplazado por named tuple, pero no se quiere depender de python 2.6
    """
    node2com = {}
    total_weight = 0
    internals = {}
    degrees = {}
    gdegrees = {}

    def __init__(self):
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Realiza una copia profunda del status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Inicializa el status de un grafo con cada nodo en una comunidad"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Grado de nodo incorrecto ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Tipo de grafo incorrecto ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    Todos los derechos reservados.
#    Licencia BSD.

__PASS_MAX = -1
__MIN = 0.0000001


def check_random_state(seed):
    """Convierte seed en una instancia de np.random.RandomState.

    Parámetros
    ----------
    seed : None | int | instancia de RandomState
        Si seed es None, retorna el singleton RandomState usado por np.random.
        Si seed es un int, retorna una nueva instancia RandomState con esa semilla.
        Si seed ya es una instancia de RandomState, la retorna.
        De lo contrario, lanza ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r no puede ser usado para inicializar un numpy.random.RandomState" % seed)


def partition_at_level(dendrogram, level):
    """Retorna la partición de los nodos en el nivel dado

    Un dendrograma es un árbol y cada nivel es una partición de los nodos del grafo.
    El nivel 0 es la primera partición, que contiene las comunidades más pequeñas,
    y la mejor es len(dendrogram) - 1.
    Cuanto mayor es el nivel, más grandes son las comunidades

    Parámetros
    ----------
    dendrogram : lista de dict
       una lista de particiones, es decir, diccionarios donde las claves de i+1 son los
       valores de i.
    level : int
       el nivel que pertenece a [0..len(dendrogram)-1]

    Retorna
    -------
    partition : diccionario
       Un diccionario donde las claves son los nodos y los valores el conjunto al que
       pertenecen

    Lanza
    ----
    KeyError
       Si el dendrograma no está bien formado o el nivel es muy alto

    Ver también
    -----------
    best_partition : que combina directamente partition_at_level y
    generate_dendrogram : para obtener la partición de mayor modularidad

    Ejemplos
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Calcula la modularidad de una partición de un grafo

    Parámetros
    ----------
    partition : dict
       la partición de los nodos, es decir, un diccionario donde las claves son los nodos
       y los valores las comunidades
    graph : networkx.Graph
       el grafo de networkx que se descompone
    weight : str, opcional
        la clave en el grafo a usar como peso. Por defecto 'weight'

    Retorna
    -------
    modularity : float
       La modularidad

    Lanza
    ------
    KeyError
       Si la partición no es una partición de todos los nodos del grafo
    ValueError
        Si el grafo no tiene enlaces
    TypeError
        Si el grafo no es un networkx.Graph

    Referencias
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Ejemplos
    --------
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)
    >>> modularity(partition, G)
    """
    if graph.is_directed():
        raise TypeError("Tipo de grafo incorrecto, use solo grafos no dirigidos")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("Un grafo sin enlaces tiene una modularidad indefinida")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None):
    """Calcula la partición de los nodos del grafo que maximiza la modularidad
    (o lo intenta...) usando la heurística de Louvain

    Esta es la partición de mayor modularidad, es decir, la partición más alta
    del dendrograma generado por el algoritmo de Louvain.

    Parámetros
    ----------
    graph : networkx.Graph
       el grafo de networkx que se descompone
    partition : dict, opcional
       el algoritmo comenzará usando esta partición de los nodos.
       Es un diccionario donde las claves son los nodos y los valores las comunidades
    weight : str, opcional
        la clave en el grafo a usar como peso. Por defecto 'weight'
    resolution :  double, opcional
        Cambia el tamaño de las comunidades, por defecto 1.
        representa el tiempo descrito en
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : booleano, opcional
        Aleatoriza el orden de evaluación de nodos y comunidades para obtener
        diferentes particiones en cada llamada
    random_state : int, instancia de RandomState o None, opcional (por defecto=None)
        Si es int, random_state es la semilla usada por el generador de números aleatorios;
        Si es instancia de RandomState, random_state es el generador de números aleatorios;
        Si es None, el generador de números aleatorios es la instancia RandomState usada
        por `np.random`.

    Retorna
    -------
    partition : diccionario
       La partición, con comunidades numeradas de 0 al número de comunidades

    Lanza
    ------
    NetworkXError
       Si el grafo no es no dirigido.

    Ver también
    -----------
    generate_dendrogram : para obtener todos los niveles de descomposición

    Notas
    -----
    Usa el algoritmo de Louvain

    Referencias
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).

    Ejemplos
    --------
    >>> # uso básico
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partion = community_louvain.best_partition(G)

    >>> # mostrar un grafo con sus comunidades:
    >>> # como los grafos Erdos-Renyi no tienen estructura real de comunidad,
    >>> # en su lugar carga el grafo del club de karate
    >>> import community as community_louvain
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # calcula la mejor partición
    >>> partition = community_louvain.best_partition(G)

    >>> # dibuja el grafo
    >>> pos = nx.spring_layout(G)
    >>> # colorea los nodos según su partición
    >>> cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    >>> nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    >>>                        cmap=cmap, node_color=list(partition.values()))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()
    """
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)

def export_communities_html(communities, output_file="3_community.html"):
    """
    Genera una vista HTML simple mostrando las comunidades detectadas.
    communities: dict {id_comunidad: [lista de nodos]}
    output_file: nombre del archivo HTML de salida
    """
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Comunidades Detectadas</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f6fa; color: #222; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 60%; margin: 30px auto; background: #fff; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background: #3498db; color: #fff; }
            tr:nth-child(even) { background: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">Comunidades Detectadas</h1>
        <table>
            <tr>
                <th>ID Comunidad</th>
                <th>Tamaño</th>
            </tr>
    """

    for comm_id, nodes in communities.items():
        html += f"<tr><td>{comm_id}</td><td>{len(nodes)}</td></tr>\n"

    html += """
        </table>
        <p style="text-align:center;">Total de comunidades: <b>{}</b></p>
    </body>
    </html>
    """.format(len(communities))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Vista HTML de comunidades exportada a {output_file}")

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    """Encuentra comunidades en el grafo y retorna el dendrograma asociado

    Un dendrograma es un árbol y cada nivel es una partición de los nodos del grafo.
    El nivel 0 es la primera partición, que contiene las comunidades más pequeñas,
    y la mejor es len(dendrogram) - 1. Cuanto mayor es el nivel, más grandes
    son las comunidades

    Parámetros
    ----------
    graph : networkx.Graph
        el grafo de networkx que será descompuesto
    part_init : dict, opcional
        el algoritmo comenzará usando esta partición de los nodos. Es un
        diccionario donde las claves son los nodos y los valores las comunidades
    weight : str, opcional
        la clave en el grafo a usar como peso. Por defecto 'weight'
    resolution :  double, opcional
        Cambia el tamaño de las comunidades, por defecto 1.
        representa el tiempo descrito en
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Retorna
    -------
    dendrogram : lista de diccionarios
        una lista de particiones, es decir, diccionarios donde las claves de i+1 son los
        valores de i. y donde las claves del primero son los nodos del grafo

    Lanza
    ------
    TypeError
        Si el grafo no es un networkx.Graph

    Ver también
    -----------
    best_partition

    Notas
    -----
    Usa el algoritmo de Louvain

    Referencias
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).

    Ejemplos
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    if graph.is_directed():
        raise TypeError("Tipo de grafo incorrecto, use solo grafos no dirigidos")

    # Manejo adecuado del estado aleatorio, eventualmente eliminar el viejo parámetro `randomize`
    # NOTA: cuando se elimine `randomize`, eliminar el código hasta random_state = ...
    if randomize is not None:
        warnings.warn("El parámetro `randomize` será deprecado en futuras versiones. Usa `random_state` en su lugar.", DeprecationWarning)
        # Si no se debe aleatorizar, se fija una semilla para obtener resultados deterministas
        if randomize is False:
            random_state = 0

    # No se sabe qué hacer si ambos `randomize` y `random_state` están definidos
    if randomize and random_state is not None:
        raise ValueError("`randomize` y `random_state` no pueden usarse al mismo tiempo")

    random_state = check_random_state(random_state)

    # caso especial, cuando no hay enlaces
    # la mejor partición es cada uno en su comunidad
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_state)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        status.init(current_graph, weight)
    return status_list[:]


def induced_graph(partition, graph, weight="weight"):
    """Produce el grafo donde los nodos son las comunidades

    hay un enlace de peso w entre comunidades si la suma de los pesos
    de los enlaces entre sus elementos es w

    Parámetros
    ----------
    partition : dict
       un diccionario donde las claves son los nodos del grafo y los valores la parte a la que
       pertenece el nodo
    graph : networkx.Graph
        el grafo inicial
    weight : str, opcional
        la clave en el grafo a usar como peso. Por defecto 'weight'

    Retorna
    -------
    g : networkx.Graph
       un grafo de networkx donde los nodos son las partes

    Ejemplos
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumera los valores del diccionario de 0 a n"""
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no es necesario renumerar
        ret = dictionary.copy()
    else:
        # agrega los valores que no serán renumerados
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # agrega los valores que serán renumerados
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Carga un grafo binario como el usado por la implementación en cpp de este algoritmo"""
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution, random_state):
    """Calcula un nivel de comunidades"""
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)
            remove_cost = - neigh_communities.get(com_node,0) + \
                resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                incr = remove_cost + dnc - \
                       resolution * status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """
    Calcula las comunidades en el vecindario de un nodo en el grafo dado
    con la descomposición node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status):
    """ Quita el nodo de la comunidad com y modifica el status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Inserta el nodo en la comunidad y modifica el status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """
    Calcula rápidamente la modularidad de la partición del grafo usando
    el status precomputado
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    """Retorna una lista con una permutación aleatoria de items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items