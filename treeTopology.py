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

    def _add_irregular_edges(self):
        for i in range(self.num_clusters - 1):
            curr_base = i * self.processors_in_cluster
            next_base = (i + 1) * self.processors_in_cluster
            self.adjacency_matrix[curr_base + 3, next_base + 4] = 1
            self.adjacency_matrix[next_base + 4, curr_base + 3] = 1


class TreeTopologyVisualizer(BaseNetworkVisualizer):
    def __init__(self, adjacency_matrix, num_clusters):
        super().__init__(adjacency_matrix, num_clusters, PROCESSORS_IN_CLUSTER, BASE_CLUSTER_POSITIONS)

    def _calculate_positions(self):
        depth = self.num_clusters.bit_length()
        LEAF_DX = 8
        LEVEL_DY = 14

        for cluster_idx in range(self.num_clusters):
            level = (cluster_idx + 1).bit_length() - 1
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

        internal_edges, regular_edges = [], []
        irregular_arcs, irregular_lines = [], []

        for u, v in self.graph.edges():
            cluster_u = (u - 1) // self.processors_in_cluster
            cluster_v = (v - 1) // self.processors_in_cluster

            if cluster_u == cluster_v:
                internal_edges.append((u, v))
            elif abs(cluster_u - cluster_v) >= 1:
                u_local = (u - 1) % self.processors_in_cluster + 1
                v_local = (v - 1) % self.processors_in_cluster + 1

                if (u_local in [7, 8] and v_local == 1) or (v_local in [7, 8] and u_local == 1):
                    regular_edges.append((u, v))
                else:
                    level_u = (cluster_u + 1).bit_length()
                    level_v = (cluster_v + 1).bit_length()

                    if level_u == level_v:
                        irregular_arcs.append((u, v))
                    else:
                        irregular_lines.append((u, v))

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=internal_edges, edge_color='gray', width=1.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=regular_edges, edge_color='#0055ff', width=2.5)

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=irregular_arcs,
                               edge_color='#2ca02c', width=2.0, style='dashed',
                               connectionstyle="arc3,rad=0.2",        # Малюємо прямі для переходів між рівнями
                               arrows=True, arrowstyle='-', node_size=400)

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=irregular_lines,
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

    csv_filename = "scaling_table_lab2.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\nТаблицю успішно збережено у файл: {csv_filename}")

    network = TreeTopologyNetwork(7)
    visualizer = TreeTopologyVisualizer(network.build_network(), 7)
    visualizer.draw()