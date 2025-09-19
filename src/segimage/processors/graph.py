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
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

from igraph import Graph
from scipy.sparse import lil_matrix
from skimage import color

from . import register_processor
from ..utils import compute_lbp_float_from_rgb_uint8


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


def graph_run(
    input_path: Path,
    output_path: Path,
    *,
    edge_filter: Optional[str] = None,
    edge_similarity: Optional[float] = None,
) -> bool:
    try:
        array, is_rgb = _load_image_as_array(input_path)
        if is_rgb:
            h, w, _ = array.shape
        else:
            h, w = array.shape

        # Normalize filter options
        filter_kind: Optional[str] = None
        if edge_filter is not None:
            ef = edge_filter.strip().lower()
            if ef in ("none", ""):
                filter_kind = None
            elif ef in ("lbp_eq", "lbp", "gray", "rgb"):
                filter_kind = ef
            else:
                raise ValueError(f"Unsupported edge_filter: {edge_filter}")
        similarity_value: float = 0.0 if edge_similarity is None else float(edge_similarity)
        if similarity_value < 0.0 or similarity_value > 1.0:
            raise ValueError("edge_similarity must be within [0, 1]")

        # Precompute features as needed by the requested filter
        lbp_code_flat: Optional[np.ndarray] = None
        lbp_code_2d: Optional[np.ndarray] = None
        gray_u8_flat: Optional[np.ndarray] = None
        gray_u8_2d: Optional[np.ndarray] = None
        rgb_u8_flat: Optional[np.ndarray] = None
        rgb_u8_3d: Optional[np.ndarray] = None

        def compute_gray_u8(a: np.ndarray, rgb: bool) -> np.ndarray:
            if rgb:
                r = a[:, :, 0].astype(np.float64)
                g = a[:, :, 1].astype(np.float64)
                b = a[:, :, 2].astype(np.float64)
                gray_f = (0.299 * r + 0.587 * g + 0.114 * b)
                return np.clip(gray_f.round(), 0, 255).astype(np.uint8)
            return a.astype(np.uint8)

        if filter_kind in ("lbp_eq", "lbp"):
            if is_rgb:
                rgb_img = array
            else:
                rgb_img = np.stack([array] * 3, axis=-1)
            lbp_float = compute_lbp_float_from_rgb_uint8(rgb_img)
            lbp_code_2d = (lbp_float * 255.0 + 0.5).astype(np.uint8)
            lbp_code_flat = lbp_code_2d.reshape(-1)

        if filter_kind == "gray":
            gray_u8_2d = compute_gray_u8(array, is_rgb)
            gray_u8_flat = gray_u8_2d.reshape(-1)

        if filter_kind == "rgb":
            if is_rgb:
                rgb = array.astype(np.uint8)
            else:
                rgb = np.stack([array] * 3, axis=-1).astype(np.uint8)
            rgb_u8_3d = rgb
            rgb_u8_flat = rgb.reshape(-1, 3)

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

                    def ok_edge(px1_y: int, px1_x: int, px2_y: int, px2_x: int) -> bool:
                        if filter_kind is None:
                            return True
                        # threshold derived from similarity: 1.0 → exact; 0.0 → always ok
                        threshold = 1.0 - similarity_value
                        if filter_kind == "lbp_eq":
                            # exact code equality only
                            assert lbp_code_2d is not None
                            return bool(lbp_code_2d[px1_y, px1_x] == lbp_code_2d[px2_y, px2_x])
                        if filter_kind == "lbp":
                            assert lbp_code_2d is not None
                            d = abs(int(lbp_code_2d[px1_y, px1_x]) - int(lbp_code_2d[px2_y, px2_x])) / 255.0
                            return bool(d <= threshold)
                        if filter_kind == "gray":
                            assert gray_u8_2d is not None
                            d = abs(int(gray_u8_2d[px1_y, px1_x]) - int(gray_u8_2d[px2_y, px2_x])) / 255.0
                            return bool(d <= threshold)
                        if filter_kind == "rgb":
                            assert rgb_u8_3d is not None
                            c1 = rgb_u8_3d[px1_y, px1_x, :].astype(np.int32)
                            c2 = rgb_u8_3d[px2_y, px2_x, :].astype(np.int32)
                            diff = c1 - c2
                            dist = float(np.sqrt(int(diff[0]) ** 2 + int(diff[1]) ** 2 + int(diff[2]) ** 2))
                            max_dist = 255.0 * np.sqrt(3.0)
                            d = dist / max_dist
                            return bool(d <= threshold)
                        return True

                    # Right neighbor
                    if x + 1 < w and ok_edge(y, x, y, x + 1):
                        nx = (x + 1) * cell_size + half
                        col = avg_color(y, x, y, x + 1)
                        draw.line([(cx, cy), (nx, cy)], fill=col, width=1)
                    # Down neighbor
                    if y + 1 < h and ok_edge(y, x, y + 1, x):
                        ny = (y + 1) * cell_size + half
                        col = avg_color(y, x, y + 1, x)
                        draw.line([(cx, cy), (cx, ny)], fill=col, width=1)
                    # Down-right neighbor
                    if x + 1 < w and y + 1 < h and ok_edge(y, x, y + 1, x + 1):
                        nx = (x + 1) * cell_size + half
                        ny = (y + 1) * cell_size + half
                        col = avg_color(y, x, y + 1, x + 1)
                        draw.line([(cx, cy), (nx, ny)], fill=col, width=1)
                    # Down-left neighbor
                    if x - 1 >= 0 and y + 1 < h and ok_edge(y, x, y + 1, x - 1):
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
        if edges.size > 0 and filter_kind is not None:
            if filter_kind == "lbp_eq":
                assert lbp_code_flat is not None
                mask = lbp_code_flat[edges[:, 0]] == lbp_code_flat[edges[:, 1]]
                edges = edges[mask]
            elif filter_kind == "lbp":
                assert lbp_code_flat is not None
                threshold = 1.0 - similarity_value
                d = np.abs(lbp_code_flat[edges[:, 0]].astype(np.int16) - lbp_code_flat[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
                edges = edges[d <= threshold]
            elif filter_kind == "gray":
                assert gray_u8_flat is not None
                threshold = 1.0 - similarity_value
                d = np.abs(gray_u8_flat[edges[:, 0]].astype(np.int16) - gray_u8_flat[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
                edges = edges[d <= threshold]
            elif filter_kind == "rgb":
                assert rgb_u8_flat is not None
                threshold = 1.0 - similarity_value
                c1 = rgb_u8_flat[edges[:, 0]].astype(np.int16)
                c2 = rgb_u8_flat[edges[:, 1]].astype(np.int16)
                diff = c1 - c2
                dist = np.sqrt((diff[:, 0].astype(np.float64) ** 2) + (diff[:, 1].astype(np.float64) ** 2) + (diff[:, 2].astype(np.float64) ** 2))
                max_dist = 255.0 * np.sqrt(3.0)
                d = dist / max_dist
                edges = edges[d <= threshold]
        if edges.size > 0:
            g.add_edges(edges.tolist())

        _save_graph(g, output_path)
        return True
    except Exception as e:
        print(f"Error running graph processor: {e}")
        return False


def build_affinity_graph_from_array(
    array: np.ndarray,
    is_rgb: bool,
    *,
    radius: int = 5,
    sigma_I: float = 10.0,
    sigma_X: float = 8.0,
) -> Graph:
    """Construct an undirected pixel affinity graph using Gaussian weights.

    Each pixel is a node. For pairs of pixels within Euclidean spatial distance
    < radius, add an edge with weight:

        w_ij = exp(-||F(i)-F(j)||^2 / sigma_I^2) * exp(-||X(i)-X(j)||^2 / sigma_X^2)

    where F(i) are pixel features (L*a*b* for RGB or intensity for grayscale)
    and X(i) are pixel coordinates (row, col).

    Returns an igraph Graph with an edge attribute "weight" and graph attrs
    "width", "height". Also sets per-vertex color attributes similar to the
    existing pixel graph (r,g,b or gray).
    """
    if array.ndim == 3 and array.shape[-1] >= 3:
        h, w, _ = array.shape
    else:
        h, w = array.shape
    num_vertices = int(h * w)

    # Prepare feature vectors F(i)
    if is_rgb:
        # skimage.color.rgb2lab expects floats in [0,1]
        rgb01 = (array[:, :, :3].astype(np.float64) / 255.0)
        lab = color.rgb2lab(rgb01)
        pixel_features = lab.reshape(num_vertices, 3).astype(np.float64)
    else:
        gray = array.astype(np.float64)
        pixel_features = gray.reshape(num_vertices, 1)

    # Spatial coordinates X(i) as (row, col)
    indices = np.arange(num_vertices, dtype=np.int64)
    pixel_coords = np.column_stack((indices // w, indices % w)).astype(np.float64)

    # Sparse matrix to assemble weights
    W = lil_matrix((num_vertices, num_vertices), dtype=np.float32)

    rad = int(max(1, radius))
    sigI = float(max(1e-12, sigma_I))
    sigX = float(max(1e-12, sigma_X))

    for i in range(num_vertices):
        r_i = int(pixel_coords[i, 0])
        c_i = int(pixel_coords[i, 1])

        r_min = max(0, r_i - rad)
        r_max = min(h, r_i + rad + 1)
        c_min = max(0, c_i - rad)
        c_max = min(w, c_i + rad + 1)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                j = r * w + c
                if i >= j:
                    continue
                # Spatial distance (row, col)
                dr = float(r - r_i)
                dc = float(c - c_i)
                dist_spatial = np.hypot(dr, dc)
                if dist_spatial >= rad:
                    continue
                # Feature distance
                df = pixel_features[i] - pixel_features[j]
                dist_feature_sq = float(np.dot(df, df))
                # Gaussian weights
                weight_I = np.exp(-(dist_feature_sq) / (sigI * sigI))
                weight_X = np.exp(-(dist_spatial * dist_spatial) / (sigX * sigX))
                weight = float(weight_I * weight_X)
                if weight <= 0.0:
                    continue
                W[i, j] = weight
                W[j, i] = weight

    # Build igraph from sparse matrix (use only upper triangle to avoid dupes)
    coo = W.tocoo()
    edges = []
    weights = []
    for i_row, j_col, w_val in zip(coo.row, coo.col, coo.data):
        if i_row < j_col:
            edges.append((int(i_row), int(j_col)))
            weights.append(float(w_val))

    g = Graph()
    g.add_vertices(num_vertices)
    g["width"] = int(w)
    g["height"] = int(h)
    if is_rgb:
        g.vs["r"] = array[:, :, 0].reshape(-1).astype(int).tolist()
        g.vs["g"] = array[:, :, 1].reshape(-1).astype(int).tolist()
        g.vs["b"] = array[:, :, 2].reshape(-1).astype(int).tolist()
    else:
        g.vs["gray"] = array.reshape(-1).astype(int).tolist()
    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights
    return g


# Register the processor
register_processor("graph", graph_run)



