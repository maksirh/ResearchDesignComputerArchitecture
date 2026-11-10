import math

import matplotlib.pyplot as plt
import networkx as nx
from core import BaseClusterNetwork, BaseNetworkVisualizer, NetworkAnalyzer

PROCESSORS_IN_CLUSTER = 8

BASE_CLUSTER_POSITIONS = {
    1: (1, 3), 2: (3, 3),
    3: (0, 2), 4: (2, 2),
    5: (1, 1), 6: (3, 1),
    7: (0, 0), 8: (2, 0),
}

BASE_INTERNAL_EDGES = [
    (1, 2), (1, 3), (1, 5), (1, 8),
    (2, 4), (2, 6), (2, 7),
    (3, 4), (3, 7), (3, 6),
    (4, 5), (4, 8),
    (5, 6), (5, 7),
    (6, 8),
    (7, 8),
]

FRAME_EDGES = {
    (2, 7): [(2.6, 3.5), (-0.5, 3.5), (-0.5, 1.2)],
    (6, 3): [(2.6, -0.5), (-0.7, -0.5), (-0.7, 0.8)],
}

REGULAR_RIGHT = [(6, 3)]
REGULAR_DOWN = [(8, 1)]

IRREGULAR_DIAG = [(4, 5)]

CLUSTER_DX = 6
CLUSTER_DY = 6


def grid_columns(num_clusters):
    return math.ceil(math.sqrt(num_clusters))


class LatticeTopologyNetwork(BaseClusterNetwork):
    def __init__(self, num_clusters, cols=None):
        super().__init__(num_clusters, PROCESSORS_IN_CLUSTER, BASE_INTERNAL_EDGES)
        self.cols = cols or grid_columns(num_clusters)

    def _cluster_index(self, row, col):
        if col >= self.cols or col < 0 or row < 0:
            return None
        idx = row * self.cols + col
        return idx if idx < self.num_clusters else None

    def _connect(self, cluster_a, cluster_b, pairs):
        if cluster_b is None:
            return
        base_a = cluster_a * self.processors_in_cluster
        base_b = cluster_b * self.processors_in_cluster
        for u, v in pairs:
            self.adjacency_matrix[base_a + u - 1, base_b + v - 1] = 1
            self.adjacency_matrix[base_b + v - 1, base_a + u - 1] = 1

    def _add_regular_edges(self):
        for i in range(self.num_clusters):
            row, col = divmod(i, self.cols)
            self._connect(i, self._cluster_index(row, col + 1), REGULAR_RIGHT)
            self._connect(i, self._cluster_index(row + 1, col), REGULAR_DOWN)

    def _add_irregular_edges(self):
        for i in range(self.num_clusters):
            row, col = divmod(i, self.cols)
            self._connect(i, self._cluster_index(row + 1, col + 1), IRREGULAR_DIAG)


class LatticeTopologyVisualizer(BaseNetworkVisualizer):
    def __init__(self, adjacency_matrix, num_clusters, cols=None):
        super().__init__(adjacency_matrix, num_clusters, PROCESSORS_IN_CLUSTER, BASE_CLUSTER_POSITIONS)
        self.cols = cols or grid_columns(num_clusters)

    def _cluster_origin(self, cluster_idx):
        row, col = divmod(cluster_idx, self.cols)
        return col * CLUSTER_DX, -row * CLUSTER_DY

    def _calculate_positions(self):
        for cluster_idx in range(self.num_clusters):
            base_x, base_y = self._cluster_origin(cluster_idx)
            base_index = cluster_idx * self.processors_in_cluster
            for local_id, (local_x, local_y) in self.base_positions.items():
                self.pos[base_index + local_id] = (base_x + local_x, base_y + local_y)

    def draw(self):
        self._calculate_positions()
        rows = math.ceil(self.num_clusters / self.cols)
        plt.figure(figsize=(max(8, self.cols * 4), max(6, rows * 4)))

        p = self.processors_in_cluster
        internal_edges, frame_edges, regular_edges, irregular_edges = [], [], [], []

        for u, v in self.graph.edges():
            cu, cv = (u - 1) // p, (v - 1) // p
            lu, lv = (u - 1) % p + 1, (v - 1) % p + 1

            if cu == cv:
                if (lu, lv) in FRAME_EDGES:
                    frame_edges.append((u, v))
                elif (lv, lu) in FRAME_EDGES:
                    frame_edges.append((v, u))
                else:
                    internal_edges.append((u, v))
                continue

            (ru, cu_col), (rv, cv_col) = divmod(cu, self.cols), divmod(cv, self.cols)
            if ru != rv and cu_col != cv_col:
                irregular_edges.append((u, v))
            else:
                regular_edges.append((u, v))

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=internal_edges, edge_color='gray', width=1.5)
        for u, v in frame_edges:
            base_x, base_y = self._cluster_origin((u - 1) // p)
            waypoints = FRAME_EDGES[((u - 1) % p + 1, (v - 1) % p + 1)]
            points = [self.pos[u]] + [(base_x + x, base_y + y) for x, y in waypoints] + [self.pos[v]]
            xs, ys = zip(*points)
            plt.plot(xs, ys, color='gray', linewidth=1.5, zorder=0)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=regular_edges, edge_color='#0055ff', width=2.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=irregular_edges, edge_color='#2ca02c', width=2.0,
                               style='dashed', arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.15",
                               node_size=500)

        nx.draw_networkx_nodes(self.graph, self.pos, node_color='#aadaff', node_size=500, edgecolors='#333333',
                               linewidths=1.3)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=9, font_weight='bold')

        for cluster_idx in range(self.num_clusters):
            base_x, base_y = self._cluster_origin(cluster_idx)
            plt.text(base_x + 1.5, base_y - 1.2, f"Кластер {cluster_idx + 1}", fontsize=11, ha='center',
                     fontweight='bold', color='#004488')

        plt.title(f"Топологія 'Решітка' (Варіант 3, {self.num_clusters} кластерів)", fontsize=14, pad=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Генерація таблиці масштабування (решітка k×k, до 1000 вузлів)...")
    max_side = math.isqrt(1000 // PROCESSORS_IN_CLUSTER)  # 11 -> 121 кластер, 968 вузлів
    df = NetworkAnalyzer.generate_scaling_table(
        None, LatticeTopologyNetwork,
        cluster_counts=[k * k for k in range(1, max_side + 1)],
    )
    print(df.to_string(index=False))

    csv_filename = "scaling_table_lattice.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nТаблицю збережено у файл: {csv_filename}")

    network = LatticeTopologyNetwork(16)
    visualizer = LatticeTopologyVisualizer(network.build_network(), 16)
    visualizer.draw()
