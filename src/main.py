import cv2
import numpy as np
import igraph as ig
from skimage.segmentation import slic
from skimage.util import img_as_float


def preprocess_image(image_path):
    """Load and convert the image to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_superpixels(image, n_segments=280, compactness=2):
    """Generate superpixels using SLIC."""
    image_float = img_as_float(image)
    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=1,
        channel_axis=-1)
    return segments


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


def img2igraph(image_path: str) -> ig.Graph:
    img = preprocess_image(image_path)
    segments = create_superpixels(img)
    graph = create_graph_from_superpixels(img, segments)
    return graph


def save_graph_as_image(graph, file_path):
    """Save a basic visualization of the graph."""
    layout = graph.layout("fr")
    ig.plot(
        graph,
        target=file_path,
        layout=layout,
        vertex_size=20,
        vertex_label=None,
        edge_width=0.5,
        edge_color="gray",
        vertex_color="red",
    )
    print(f"Graph image saved to {file_path}")


def main():
    image_path = '5096.jpg'
    graph = img2igraph(f'images/original/{image_path}')
    save_graph_as_image(graph, f'images/output/{image_path.replace(".jpg", ".png")}')
    return True


__name__ == '__main__' and main()