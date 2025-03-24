from .preprocess import preprocess_image
from .superpixel import create_superpixels
from .graph import create_graph_from_superpixels

def image_to_igraph(image_path: str):
    img = preprocess_image(image_path)
    segments = create_superpixels(img)
    graph = create_graph_from_superpixels(img, segments)
    return graph

