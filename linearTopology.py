import matplotlib.pyplot as plt
import networkx as nx
from core import BaseClusterNetwork, BaseNetworkVisualizer, NetworkAnalyzer

PROCESSORS_IN_CLUSTER = 6
CLUSTER_OFFSET_X = 6

BASE_CLUSTER_POSITIONS = {
    1: (1, 2.5), 2: (0, 1.25), 3: (2, 1.25),
    4: (-1, 0), 5: (1, 0), 6: (3, 0)
}

BASE_INTERNAL_EDGES = [
    (1, 2), (1, 3), (2, 3),
    (2, 4), (2, 5), (3, 5), (3, 6),
    (4, 5), (5, 6)
]


class LinearTopologyNetwork(BaseClusterNetwork):
    def __init__(self, num_clusters):
        super().__init__(num_clusters, PROCESSORS_IN_CLUSTER, BASE_INTERNAL_EDGES)

    def _add_regular_edges(self):
        for i in range(self.num_clusters - 1):
            curr_base = i * self.processors_in_cluster
            next_base = (i + 1) * self.processors_in_cluster

            self.adjacency_matrix[curr_base + 5, next_base + 3] = 1
            self.adjacency_matrix[next_base + 3, curr_base + 5] = 1

            self.adjacency_matrix[curr_base + 2, next_base + 1] = 1
            self.adjacency_matrix[next_base + 1, curr_base + 2] = 1

    def _add_irregular_edges(self):
        for i in range(self.num_clusters - 2):
            curr_base = i * self.processors_in_cluster
            jump_base = (i + 2) * self.processors_in_cluster

            self.adjacency_matrix[curr_base + 0, jump_base + 0] = 1
            self.adjacency_matrix[jump_base + 0, curr_base + 0] = 1

            self.adjacency_matrix[curr_base + 4, jump_base + 4] = 1
            self.adjacency_matrix[jump_base + 4, curr_base + 4] = 1


class LinearTopologyVisualizer(BaseNetworkVisualizer):
    def __init__(self, adjacency_matrix, num_clusters):
        super().__init__(adjacency_matrix, num_clusters, PROCESSORS_IN_CLUSTER, BASE_CLUSTER_POSITIONS)

    def _calculate_positions(self):
        for cluster_num in range(self.num_clusters):
            base_x = cluster_num * CLUSTER_OFFSET_X
            base_index = cluster_num * self.processors_in_cluster
            for local_id, (local_x, local_y) in self.base_positions.items():
                self.pos[base_index + local_id] = (base_x + local_x, local_y)

    def draw(self):
        self._calculate_positions()
        fig_width = max(10, self.num_clusters * 3.5)
        plt.figure(figsize=(fig_width, 6))

        internal_edges, regular_edges, top_jumps, bottom_jumps = [], [], [], []

        for u, v in self.graph.edges():
            cluster_u = (u - 1) // self.processors_in_cluster
            cluster_v = (v - 1) // self.processors_in_cluster
            diff = abs(cluster_u - cluster_v)

            if diff == 0:
                internal_edges.append((u, v))
            elif diff == 1:
                regular_edges.append((u, v))
            else:
                local_u = (u - 1) % self.processors_in_cluster + 1
                if local_u == 1:
                    top_jumps.append((u, v))
                else:
                    bottom_jumps.append((u, v))

        nx.draw_networkx_edges(self.graph, self.pos, edgelist=internal_edges, edge_color='gray', width=1.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=regular_edges, edge_color='#0055ff', width=2.5)
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=top_jumps, edge_color='#2ca02c', width=2.0,
                               style='dashed', arrows=True, arrowstyle='-', connectionstyle="arc3,rad=-0.25")
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=bottom_jumps, edge_color='#2ca02c', width=2.0,
                               style='dashed', arrows=True, arrowstyle='-', connectionstyle="arc3,rad=0.25")

        nx.draw_networkx_nodes(self.graph, self.pos, node_color='#aadaff', node_size=700, edgecolors='#333333',
                               linewidths=1.5)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=10, font_weight='bold')

        for cluster_num in range(self.num_clusters):
            base_x = cluster_num * CLUSTER_OFFSET_X
            plt.text(base_x + 1, -1.5, f"Кластер {cluster_num + 1}", fontsize=12, ha='center', fontweight='bold',
                     color='#004488')

        plt.title(f"Лінійна топологія (Варіант 4)", fontsize=14, pad=15)
        plt.axis('off')
        plt.ylim(-2.5, 4.0)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Генерація таблиці
    print("Генерація таблиці масштабування (ЛР 1)...")
    df = NetworkAnalyzer.generate_scaling_table(17, LinearTopologyNetwork)
    print(df.to_string(index=False))

    # Візуалізація
    network = LinearTopologyNetwork(4)
    visualizer = LinearTopologyVisualizer(network.build_network(), 4)
    visualizer.draw()