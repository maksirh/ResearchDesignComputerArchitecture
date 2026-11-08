from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pandas as pd


class BaseClusterNetwork(ABC):
    def __init__(self, num_clusters, processors_in_cluster, base_internal_edges):
        self.num_clusters = num_clusters
        self.processors_in_cluster = processors_in_cluster
        self.base_internal_edges = base_internal_edges
        self.total_nodes = num_clusters * processors_in_cluster
        self.adjacency_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=int)

    def build_network(self):
        self._add_internal_edges()
        self._add_regular_edges()
        self._add_irregular_edges()
        return self.adjacency_matrix

    def _add_internal_edges(self):
        for i in range(self.num_clusters):
            base = i * self.processors_in_cluster
            for u, v in self.base_internal_edges:
                self.adjacency_matrix[base + u - 1, base + v - 1] = 1
                self.adjacency_matrix[base + v - 1, base + u - 1] = 1

    @abstractmethod
    def _add_regular_edges(self):
        pass

    @abstractmethod
    def _add_irregular_edges(self):
        pass


class BaseNetworkVisualizer(ABC):
    def __init__(self, adjacency_matrix, num_clusters, processors_in_cluster, base_positions):
        self.matrix = adjacency_matrix
        self.num_clusters = num_clusters
        self.processors_in_cluster = processors_in_cluster
        self.base_positions = base_positions
        self.total_nodes = num_clusters * processors_in_cluster
        self.graph = nx.Graph()
        self.pos = {}
        self._build_graph()

    def _build_graph(self):
        for i in range(self.total_nodes):
            self.graph.add_node(i + 1)
        for i in range(self.total_nodes):
            for j in range(i + 1, self.total_nodes):
                if self.matrix[i, j] == 1:
                    self.graph.add_edge(i + 1, j + 1)

    @abstractmethod
    def _calculate_positions(self):
        pass

    @abstractmethod
    def draw(self):
        pass


class NetworkAnalyzer:
    def __init__(self, adjacency_matrix):
        self.matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        self.graph = nx.from_numpy_array(self.matrix)

    def analyze(self):
        degrees = [deg for node, deg in self.graph.degree()]
        S = max(degrees) if degrees else 0
        shortest_paths = nx.floyd_warshall_numpy(self.graph)
        D = int(np.max(shortest_paths)) if self.num_nodes > 0 else 0

        if self.num_nodes > 1:
            total_paths_sum = np.sum(shortest_paths)
            aD = total_paths_sum / (self.num_nodes * (self.num_nodes - 1))
        else:
            aD = 0

        C = D * S
        T = 2 * aD * S

        return {
            "Кількість вузлів (N)": self.num_nodes,
            "Ступінь (S)": S,
            "Діаметр (D)": D,
            "Середній діаметр (aD)": round(aD, 3),
            "Вартість (C)": C,
            "Трафік (T)": round(T, 3)
        }

    @staticmethod
    def generate_scaling_table(max_clusters, network_class, cluster_step=1):
        results = []

        counts = [1] + list(range(cluster_step, max_clusters + 1, cluster_step))
        counts = sorted(list(set(counts)))

        for num_clusters in counts:
            network = network_class(num_clusters)
            matrix = network.build_network()
            analyzer = NetworkAnalyzer(matrix)
            metrics = analyzer.analyze()

            results.append({
                "N (кількість процесорів)": metrics["Кількість вузлів (N)"],
                "D": metrics["Діаметр (D)"],
                "D̄": metrics["Середній діаметр (aD)"],
                "S": metrics["Ступінь (S)"],
                "C": metrics["Вартість (C)"],
                "T": metrics["Трафік (T)"]
            })
        return pd.DataFrame(results)