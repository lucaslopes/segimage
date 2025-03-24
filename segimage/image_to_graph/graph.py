import cv2
import numpy as np
import igraph as ig


def create_graph_from_superpixels(image, segments) -> ig.Graph:
    """Build an igraph graph from superpixels based on 8-neighbor adjacency."""
    height, width = image.shape[:2]
    n_superpixels = np.max(segments)
    graph = ig.Graph()
    graph.add_vertices(n_superpixels)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    weights = np.zeros(n_superpixels, dtype=np.float32)
    counts = np.zeros(n_superpixels, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            sp_id = segments[i, j] - 1
            weights[sp_id] += gray[i, j] / 255.0
            counts[sp_id] += 1
    weights /= counts
    graph.vs["weight"] = weights
    edges = set()
    edge_weights = []
    for i in range(height):
        for j in range(width):
            sp_id = segments[i, j] - 1
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_sp_id = segments[ni, nj] - 1
                    if sp_id != neighbor_sp_id:
                        edge = tuple(sorted((sp_id, neighbor_sp_id)))
                        if edge not in edges:
                            edges.add(edge)
                            weight_diff = abs(weights[sp_id] - weights[neighbor_sp_id])
                            edge_weights.append(weight_diff)
    graph.add_edges(list(edges))
    graph.es["weight"] = edge_weights
    return graph

