from __future__ import annotations

import numpy as np
from igraph import Graph

from ..processors.graph import build_affinity_graph_from_array
from .segments import (
    compute_slico_segments,
    compute_segment_means_and_centroids,
    compute_segment_adjacency_edges,
    ensure_rgb_uint8,
)
from . import register_graph_builder


def build_affinity_pixel_graph(
    array: np.ndarray,
    is_rgb: bool,
    *,
    node_mode: str = "pixel",
    radius: int = 5,
    sigma_I: float = 10.0,
    sigma_X: float = 8.0,
    # superpixel (SLICO) parameters
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> Graph:
    mode = (node_mode or "pixel").strip().lower()
    if mode == "pixel":
        return build_affinity_graph_from_array(
            array,
            is_rgb,
            radius=int(radius),
            sigma_I=float(sigma_I),
            sigma_X=float(sigma_X),
        )

    # superpixel mode: build adjacency graph of segments with Gaussian weights
    image_u8 = ensure_rgb_uint8(array, is_rgb)
    segments = compute_slico_segments(
        array, is_rgb,
        n_segments=int(n_segments), compactness=float(compactness), sigma=float(sigma), start_label=int(start_label)
    )
    mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
    edges = compute_segment_adjacency_edges(segments)
    num_nodes = int(mean_r.size)

    # Build weights using mean features and centroid distances
    if is_rgb:
        # Convert mean RGB to Lab-like feature by simple scaling; keep it simple here
        pixel_features = np.stack([mean_r, mean_g, mean_b], axis=-1).astype(np.float64)
    else:
        pixel_features = mean_gray.reshape(-1, 1).astype(np.float64)
    coords = np.stack([cy, cx], axis=-1).astype(np.float64)

    sigI = float(max(1e-12, sigma_I))
    sigX = float(max(1e-12, sigma_X))
    weights: list[float] = []
    for u, v in edges:
        df = pixel_features[u] - pixel_features[v]
        dist_feature_sq = float(np.dot(df, df))
        dc = coords[u] - coords[v]
        dist_spatial = float(np.hypot(dc[0], dc[1]))
        w_I = float(np.exp(-(dist_feature_sq) / (sigI * sigI)))
        w_X = float(np.exp(-(dist_spatial * dist_spatial) / (sigX * sigX)))
        weights.append(float(w_I * w_X))

    h, w = segments.shape
    g = Graph()
    g.add_vertices(int(num_nodes))
    g["width"] = int(w)
    g["height"] = int(h)
    g["node_mode"] = "superpixel"
    g.vs["r"] = mean_r.astype(int).tolist()
    g.vs["g"] = mean_g.astype(int).tolist()
    g.vs["b"] = mean_b.astype(int).tolist()
    g.vs["gray"] = mean_gray.astype(int).tolist()
    g.vs["cx"] = cx.astype(float).tolist()
    g.vs["cy"] = cy.astype(float).tolist()
    if edges.size > 0:
        g.add_edges(edges.tolist())
        g.es["weight"] = weights
    return g


register_graph_builder("affinity", build_affinity_pixel_graph)


