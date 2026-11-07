import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall
import pandas as pd

PROCESSORS_IN_CLUSTER = 6
CLUSTER_OFFSET_X = 6

BASE_CLUSTER_POSITIONS = {
    1: (1, 2.5),
    2: (0, 1.25),
    3: (2, 1.25),
    4: (-1, 0),
    5: (1, 0),
    6: (3, 0)
}
BASE_INTERNAL_EDGES = [
    (1, 2), (1, 3),
    (2, 3),
    (2, 4), (2, 5),
    (3, 5), (3, 6),
    (4, 5), (5, 6)
]


class ClusterNetwork:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.total_nodes = num_clusters * PROCESSORS_IN_CLUSTER
        self.adjacency_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=int)


    def build_network(self):
        self._add_internal_edges()
        self._add_regular_edges()
        self._add_irregular_edges()
        return self.adjacency_matrix


    def _add_internal_edges(self):
        for cluster_num in range(self.num_clusters):
            base_index = cluster_num * PROCESSORS_IN_CLUSTER
            for u, v in BASE_INTERNAL_EDGES:
                node1, node2 = base_index + u - 1, base_index + v - 1
                self.adjacency_matrix[node1, node2] = 1
                self.adjacency_matrix[node2, node1] = 1

    def _add_regular_edges(self):
        for i in range(self.num_clusters - 1):
            curr_base = i * PROCESSORS_IN_CLUSTER
            next_base = (i + 1) * PROCESSORS_IN_CLUSTER

            n6, n4 = curr_base + 5, next_base + 3
            self.adjacency_matrix[n6, n4] = 1
            self.adjacency_matrix[n4, n6] = 1

            n3, n2 = curr_base + 2, next_base + 1
            self.adjacency_matrix[n3, n2] = 1
            self.adjacency_matrix[n2, n3] = 1


    def _add_irregular_edges(self):
        for i in range(self.num_clusters - 2):
            curr_base = i * PROCESSORS_IN_CLUSTER
            jump_base = (i + 2) * PROCESSORS_IN_CLUSTER

            n1_curr, n1_jump = curr_base + 0, jump_base + 0
            self.adjacency_matrix[n1_curr, n1_jump] = 1
            self.adjacency_matrix[n1_jump, n1_curr] = 1

            n5_curr, n5_jump = curr_base + 4, jump_base + 4
            self.adjacency_matrix[n5_curr, n5_jump] = 1
            self.adjacency_matrix[n5_jump, n5_curr] = 1




class NetworkVisualizer:
    def __init__(self, adjacency_matrix, num_clusters):
        self.matrix = adjacency_matrix
        self.num_clusters = num_clusters
        self.total_nodes = num_clusters * PROCESSORS_IN_CLUSTER
        self.graph = nx.Graph()
        self.pos = {}
        self._build_graph_from_matrix()

    def _build_graph_from_matrix(self):
        for i in range(self.total_nodes):
            self.graph.add_node(i + 1)

        for i in range(self.total_nodes):
            for j in range(i + 1, self.total_nodes):
                if self.matrix[i, j] == 1:
                    self.graph.add_edge(i + 1, j + 1)

    def _calculate_positions(self):
        for cluster_num in range(self.num_clusters):
            base_x = cluster_num * CLUSTER_OFFSET_X
            base_index = cluster_num * PROCESSORS_IN_CLUSTER

            for local_id, (local_x, local_y) in BASE_CLUSTER_POSITIONS.items():
                global_node_id = base_index + local_id

                self.pos[global_node_id] = (base_x + local_x, local_y)

    def draw(self):
        self._calculate_positions()

        fig_width = max(10, self.num_clusters * 3.5)
        plt.figure(figsize=(fig_width, 6))

        internal_edges, regular_edges, irregular_edges = [], [], []

        for u, v in self.graph.edges():
            cluster_u = (u - 1) // PROCESSORS_IN_CLUSTER
            cluster_v = (v - 1) // PROCESSORS_IN_CLUSTER
            diff = abs(cluster_u - cluster_v)

            if diff == 0:
                internal_edges.append((u, v))
            elif diff == 1:
                regular_edges.append((u, v))
            else:
                irregular_edges.append((u, v))

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=internal_edges,
                               edge_color='gray', width=1.5)

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=regular_edges,
                               edge_color='#0055ff', width=2.5)

        top_jumps = []
        bottom_jumps = []
        for u, v in irregular_edges:
            local_u = (u - 1) % PROCESSORS_IN_CLUSTER + 1
            if local_u == 1:
                top_jumps.append((u, v))
            else:
                bottom_jumps.append((u, v))

        nx.draw_networkx_edges(
            self.graph, self.pos, edgelist=top_jumps,
            edge_color='#2ca02c', width=2.0, style='dashed',
            arrows=True, arrowstyle='-', connectionstyle="arc3,rad=-0.25"
        )

        nx.draw_networkx_edges(
            self.graph, self.pos, edgelist=bottom_jumps,
            edge_color='#2ca02c', width=2.0, style='dashed',
            arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.25"
        )

        nx.draw_networkx_nodes(self.graph, self.pos, node_color='#aadaff',
                               node_size=700, edgecolors='#333333', linewidths=1.5)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=10, font_weight='bold')

        for cluster_num in range(self.num_clusters):
            base_x = cluster_num * CLUSTER_OFFSET_X
            plt.text(base_x + 1, -1.5, f"Кластер {cluster_num + 1}",
                     fontsize=12, ha='center', fontweight='bold', color='#004488')

        plt.title(f"Лінійна топологія (Регулярні та нерегулярні зв'язки)", fontsize=14, pad=15)
        plt.axis('off')

        plt.ylim(-2.5, 4.0)

        plt.tight_layout()
        plt.show()


class NetworkAnalyzer:
    def __init__(self, adjacency_matrix):
        self.matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        self.graph = nx.from_numpy_array(self.matrix)

    def analyze(self):
        # 1. Ступінь вузла (S)
        degrees = [deg for node, deg in self.graph.degree()]
        S = max(degrees)

        # 2. Матриця найкоротших шляхів (Алгоритм Флойда-Уоршелла)
        shortest_paths = nx.floyd_warshall_numpy(self.graph)

        # 3. Діаметр (D)
        D = int(np.max(shortest_paths))

        # 4. Середній діаметр (aD)
        total_paths_sum = np.sum(shortest_paths)
        aD = total_paths_sum / (self.num_nodes * (self.num_nodes - 1))

        # 5. Топологічна вартість (C)
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
    def generate_scaling_table(max_clusters, network_class):
        results = []

        for num_clusters in range(1, max_clusters + 1):
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


if __name__ == "__main__":
    clusters_count = 4
    max_step = 17

    # 1. Логіка
    network = ClusterNetwork(clusters_count)
    matrix = network.build_network()

    # 2. Аналіз метрик
    analyzer = NetworkAnalyzer(matrix)
    metrics = analyzer.analyze()

    print(f"\n--- Результати аналізу ({clusters_count} кластерів) ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
    print("-------------------------------------\n")

    # 3. Візуалізація
    visualizer = NetworkVisualizer(matrix, clusters_count)
    visualizer.draw()

    df_metrics = NetworkAnalyzer.generate_scaling_table(max_step, ClusterNetwork)

    print(df_metrics.to_string(index=False))

    df_metrics.to_csv("metrics.csv", index=False)


