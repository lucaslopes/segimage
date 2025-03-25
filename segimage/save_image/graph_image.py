import random
from hedonic import Game
import igraph as ig
from segimage.utils import get_partition_colors

def save_graph_as_image(graph, file_path, partition=None, seed=0):
    """Save a visualization of the graph, optionally colored by community membership."""
    g = graph.to_igraph() if isinstance(graph, Game) else graph
    layout = g.layout("fr")

    if partition is not None:
        vertex_colors = get_partition_colors(partition, seed=seed)
    else:
        vertex_colors = "red"

    ig.plot(
        g,
        target=file_path,
        layout=layout,
        vertex_size=20,
        vertex_label=None,
        edge_width=0.5,
        edge_color="gray",
        vertex_color=vertex_colors,
    )
    print(f"Graph image saved to {file_path}")
