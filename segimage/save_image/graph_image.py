import random
from hedonic import Game
import igraph as ig

def _hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def _get_partition_colors(partition, seed):
    """Return a list of colors for each community ID in the partition."""
    membership = partition.membership
    unique_communities = sorted(set(membership))
    random.seed(seed)
    color_list = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in unique_communities
    ]
    community_to_color = {
        c: color_list[i] for i, c in enumerate(unique_communities)
    }
    return [community_to_color[m] for m in membership]

def save_graph_as_image(graph, file_path, partition=None, seed=0):
    """Save a visualization of the graph, optionally colored by community membership."""
    g = graph.to_igraph() if isinstance(graph, Game) else graph
    layout = g.layout("fr")

    if partition is not None:
        vertex_colors = _get_partition_colors(partition, seed=seed)
        for idx, color in enumerate(vertex_colors):
            # Print node index, segment index, node color, segment color
            print(f"Node index: {idx}, Node color: {color}, Segment color: {_hex_to_rgb(color)}")
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
