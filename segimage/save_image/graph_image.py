import igraph as ig


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
