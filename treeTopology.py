import matplotlib.pyplot as plt
import networkx as nx
from core import BaseClusterNetwork, BaseNetworkVisualizer, NetworkAnalyzer

PROCESSORS_IN_CLUSTER = 8

BASE_CLUSTER_POSITIONS = {
    1: (0, 8),
    2: (-2, 6), 3: (2, 6),
    4: (-2, 4), 5: (2, 4),
    6: (0, 2),
    7: (-2, 0), 8: (2, 0)
}

BASE_INTERNAL_EDGES = [
    (1, 2), (1, 3), (2, 4), (3, 5),
    (4, 6), (5, 6), (6, 7), (6, 8), (7, 8)
]


def cluster_level(cluster_idx):
    return (cluster_idx + 1).bit_length() - 1


class TreeTopologyNetwork(BaseClusterNetwork):
    def __init__(self, num_clusters):
        super().__init__(num_clusters, PROCESSORS_IN_CLUSTER, BASE_INTERNAL_EDGES)

    def _add_regular_edges(self):
        for i in range(self.num_clusters):
            curr_base = i * self.processors_in_cluster
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if left_child < self.num_clusters:
                left_base = left_child * self.processors_in_cluster
                self.adjacency_matrix[curr_base + 6, left_base + 0] = 1
                self.adjacency_matrix[left_base + 0, curr_base + 6] = 1

            if right_child < self.num_clusters:
                right_base = right_child * self.processors_in_cluster

                self.adjacency_matrix[curr_base + 7, right_base + 0] = 1
                self.adjacency_matrix[right_base + 0, curr_base + 7] = 1

    def _connect(self, cluster_a, local_a, cluster_b, local_b):
        a = cluster_a * self.processors_in_cluster + local_a - 1
        b = cluster_b * self.processors_in_cluster + local_b - 1
        self.adjacency_matrix[a, b] = 1
        self.adjacency_matrix[b, a] = 1

    def _add_irregular_edges(self):
        for i in range(self.num_clusters - 1):
            right = i + 1
            if cluster_level(i) != cluster_level(right):
                continue

            nephew_right = 2 * right + 1
            if nephew_right < self.num_clusters:
                self._connect(i, 5, nephew_right, 2)

            nephew_left = 2 * i + 2
            if nephew_left < self.num_clusters:
                self._connect(right, 4, nephew_left, 3)


class TreeTopologyVisualizer(BaseNetworkVisualizer):
    def __init__(self, adjacency_matrix, num_clusters):
        super().__init__(adjacency_matrix, num_clusters, PROCESSORS_IN_CLUSTER, BASE_CLUSTER_POSITIONS)

    def _calculate_positions(self):
        depth = self.num_clusters.bit_length()
        LEAF_DX = 8
        LEVEL_DY = 14

        for cluster_idx in range(self.num_clusters):
            level = cluster_level(cluster_idx)
            pos_in_level = cluster_idx - (2 ** level - 1)

            span = LEAF_DX * (2 ** (depth - 1 - level))
            x = (pos_in_level - (2 ** level - 1) / 2) * span
            y = -level * LEVEL_DY

            base_index = cluster_idx * self.processors_in_cluster
            for local_id, (local_x, local_y) in self.base_positions.items():
                self.pos[base_index + local_id] = (x + local_x, y + local_y)

    def draw(self):
        self._calculate_positions()
        plt.figure(figsize=(16, 10))

        internal_edges, regular_edges, irregular_edges = [], [], []

        for u, v in self.graph.edges():
            cluster_u = (u - 1) // self.processors_in_cluster
            cluster_v = (v - 1) // self.processors_in_cluster

            if cluster_u == cluster_v:
                internal_edges.append((u, v))
            elif (max(cluster_u, cluster_v) - 1) // 2 == min(cluster_u, cluster_v):
                regular_edges.append((u, v))      # батько - син
            else:
                irregular_edges.append((u, v))

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=internal_edges, edge_color='gray', width=1.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=regular_edges, edge_color='#0055ff', width=2.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=irregular_edges,
                               edge_color='#2ca02c', width=2.0, style='dashed')

        nx.draw_networkx_nodes(self.graph, self.pos, node_color='#aadaff', node_size=400, edgecolors='#333333',
                               linewidths=1.2)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=8, font_weight='bold')

        plt.title(f"Топологія 'Бінарне Дерево' (Варіант 14, {self.num_clusters} кластерів)", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Генерація таблиці масштабування (ЛР 2, до 1000 вузлів)...")
    df = NetworkAnalyzer.generate_scaling_table(125, TreeTopologyNetwork, cluster_step=6)

    print(df.head(10).to_string(index=False))
    print("...\n")
    print(df.tail(10).to_string(index=False))

    excel_filename = "scaling_table_lab1.xlsx"
    df.to_excel(excel_filename, index=False, sheet_name="Масштабування ЛР1")
    print(f"Таблицю збережено у {excel_filename}\n")

    network = TreeTopologyNetwork(17)
    visualizer = TreeTopologyVisualizer(network.build_network(), 17)
    visualizer.draw()