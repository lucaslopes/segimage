"""
Graph processor using python-igraph.

Builds an undirected 8-neighbor pixel adjacency graph from an image.
Each pixel is a vertex and has edges to its 8-connected neighbors. The
graph stores useful attributes:
 - width, height: image dimensions (graph attributes)
 - For grayscale images: vertex attribute `gray` in [0,255]
 - For RGB images: vertex attributes `r`, `g`, `b` in [0,255]

Outputs are graph file formats determined by the output file suffix:
 - .graphml → GraphML
 - .gml     → GML
 - .lg, .lgl → LGL
 - .edgelist, .edges, .txt → edge list
 - .pickle, .pkl → igraph's pickled format

Note: Large images create very large graphs (O(H*W) vertices and up to
8*H*W edges). Consider downsampling prior to running this processor for
very high-resolution inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from igraph import Graph

from . import register_processor


def _load_image_as_array(input_path: Path) -> Tuple[np.ndarray, bool]:
    """Load an image and return (array, is_rgb).

    - If input is .npy, load the array directly. If 2D, treat as grayscale;
      if 3D with 3+ channels, take first 3 as RGB.
    - Otherwise load via PIL and convert to RGB. We'll also expose grayscale
      when the image is single channel.
    """
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".npy":
        arr = np.load(str(input_path))
        if arr.ndim == 2:
            # grayscale
            return arr.astype(np.uint8), False
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3].astype(np.uint8), True
        raise ValueError("Unsupported array shape for graph processor")
    img = Image.open(input_path)
    # Preserve grayscale if already L; otherwise RGB
    if img.mode == "L":
        return np.array(img, dtype=np.uint8), False
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8), True


def _build_8_neighbor_edges(height: int, width: int) -> np.ndarray:
    """Vectorized generation of 8-neighbor undirected edges for HxW grid.

    Returns an array of shape (E, 2) with vertex index pairs.
    """
    # For undirected graphs, add each pair once. We'll connect to neighbors
    # with non-negative direction displacements to avoid duplicates:
    # Right (0,+1), Down (+1,0), Down-Right (+1,+1), Down-Left (+1,-1)
    edges = []

    def vid(y: int, x: int) -> int:
        return y * width + x

    # Right
    for y in range(height):
        for x in range(width - 1):
            edges.append((vid(y, x), vid(y, x + 1)))
    # Down
    for y in range(height - 1):
        for x in range(width):
            edges.append((vid(y, x), vid(y + 1, x)))
    # Down-Right
    for y in range(height - 1):
        for x in range(width - 1):
            edges.append((vid(y, x), vid(y + 1, x + 1)))
    # Down-Left
    for y in range(height - 1):
        for x in range(1, width):
            edges.append((vid(y, x), vid(y + 1, x - 1)))

    return np.array(edges, dtype=np.int64)


def _save_graph(g: Graph, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".graphml":
        g.write_graphml(str(output_path))
        return
    if suffix == ".gml":
        g.write_gml(str(output_path))
        return
    if suffix in (".lg", ".lgl"):
        g.write_lgl(str(output_path))
        return
    if suffix in (".edgelist", ".edges", ".txt"):
        g.write_edgelist(str(output_path))
        return
    if suffix in (".pickle", ".pkl"):
        g.write_pickle(str(output_path))
        return
    # Default to GraphML if unknown
    g.write_graphml(str(output_path.with_suffix(".graphml")))


def graph_run(input_path: Path, output_path: Path) -> bool:
    try:
        array, is_rgb = _load_image_as_array(input_path)
        if is_rgb:
            h, w, _ = array.shape
        else:
            h, w = array.shape

        # If the requested output is an image format, render a grid-plot of the graph
        suffix = output_path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            # Render graph on a spaced grid so nodes don't touch.
            # Each pixel becomes a cell of size `cell_size`; the node is a smaller square
            # centered in the cell, leaving blank margins around it.
            # Choose circular nodes and set spacing so the blank space between
            # adjacent nodes is 2x the node diameter. If D is the node diameter
            # and G is the blank space, we want G = 2*D, and the center-to-center
            # distance becomes D + G = 3*D. We set cell_size = 3*D.
            node_radius = 2  # radius in pixels (default)
            node_diameter = 2 * node_radius
            cell_size = 3 * node_diameter  # ensures 2x diameter blank space
            half = cell_size // 2

            out_w = w * cell_size
            out_h = h * cell_size

            # Background: white for clean spacing
            img = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img, "RGBA")

            # Draw edges between node centers with color as the intermediate
            # (average) of the two node colors; keep edges as thin as possible.
            for y in range(h):
                for x in range(w):
                    cx = x * cell_size + half
                    cy = y * cell_size + half

                    def avg_color(px1_y: int, px1_x: int, px2_y: int, px2_x: int) -> tuple[int, int, int, int]:
                        if is_rgb:
                            c1 = array[px1_y, px1_x, :3].astype(int)
                            c2 = array[px2_y, px2_x, :3].astype(int)
                            avg = ((c1 + c2) // 2).tolist()
                            return (int(avg[0]), int(avg[1]), int(avg[2]), 255)
                        else:
                            v1 = int(array[px1_y, px1_x])
                            v2 = int(array[px2_y, px2_x])
                            v = (v1 + v2) // 2
                            return (v, v, v, 255)

                    # Right neighbor
                    if x + 1 < w:
                        nx = (x + 1) * cell_size + half
                        col = avg_color(y, x, y, x + 1)
                        draw.line([(cx, cy), (nx, cy)], fill=col, width=1)
                    # Down neighbor
                    if y + 1 < h:
                        ny = (y + 1) * cell_size + half
                        col = avg_color(y, x, y + 1, x)
                        draw.line([(cx, cy), (cx, ny)], fill=col, width=1)
                    # Down-right neighbor
                    if x + 1 < w and y + 1 < h:
                        nx = (x + 1) * cell_size + half
                        ny = (y + 1) * cell_size + half
                        col = avg_color(y, x, y + 1, x + 1)
                        draw.line([(cx, cy), (nx, ny)], fill=col, width=1)
                    # Down-left neighbor
                    if x - 1 >= 0 and y + 1 < h:
                        nx = (x - 1) * cell_size + half
                        ny = (y + 1) * cell_size + half
                        col = avg_color(y, x, y + 1, x - 1)
                        draw.line([(cx, cy), (nx, ny)], fill=col, width=1)

            # Draw nodes as small circles colored by pixel value
            for y in range(h):
                for x in range(w):
                    cx = x * cell_size + half
                    cy = y * cell_size + half
                    top_left_x = cx - node_radius
                    top_left_y = cy - node_radius
                    bottom_right_x = cx + node_radius
                    bottom_right_y = cy + node_radius
                    if is_rgb:
                        r, g, b = [int(v) for v in array[y, x, :3]]
                    else:
                        v = int(array[y, x])
                        r = g = b = v
                    draw.ellipse(
                        [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)],
                        fill=(r, g, b, 255),
                        outline=(r, g, b, 255),
                        width=1,
                    )

            # Save image in requested format
            out_mode_img = img
            if suffix in (".jpg", ".jpeg"):
                out_mode_img = img.convert("RGB")

            fmt = (
                "PNG"
                if suffix == ".png"
                else "JPEG"
                if suffix in (".jpg", ".jpeg")
                else "TIFF"
                if suffix in (".tif", ".tiff")
                else None
            )
            out_mode_img.save(output_path, fmt) if fmt else out_mode_img.save(output_path)
            return True

        # Otherwise, build and save a graph file
        num_vertices = int(h * w)
        g = Graph()
        g.add_vertices(num_vertices)
        g["width"] = int(w)
        g["height"] = int(h)

        # Vertex attributes
        if is_rgb:
            g.vs["r"] = array[:, :, 0].reshape(-1).astype(int).tolist()
            g.vs["g"] = array[:, :, 1].reshape(-1).astype(int).tolist()
            g.vs["b"] = array[:, :, 2].reshape(-1).astype(int).tolist()
        else:
            g.vs["gray"] = array.reshape(-1).astype(int).tolist()

        # Edges (8-connectivity, undirected without duplicates)
        edges = _build_8_neighbor_edges(h, w)
        if edges.size > 0:
            g.add_edges(edges.tolist())

        _save_graph(g, output_path)
        return True
    except Exception as e:
        print(f"Error running graph processor: {e}")
        return False


# Register the processor
register_processor("graph", graph_run)



