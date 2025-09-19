from __future__ import annotations

import numpy as np
from igraph import Graph

from .segments import (
    compute_slico_segments,
    compute_segment_means_and_centroids,
    compute_segment_adjacency_edges,
    ensure_rgb_uint8,
)
from . import register_graph_builder


def build_prob4_pixel_graph(
    array: np.ndarray,
    is_rgb: bool,
    *,
    node_mode: str = "pixel",
    sigma_I: float = 10.0,
    # superpixel (SLICO) parameters
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> Graph:
    mode = (node_mode or "pixel").strip().lower()
    if mode == "pixel":
        if is_rgb:
            h, w, _ = array.shape
            r = array[:, :, 0].astype(np.float64)
            g = array[:, :, 1].astype(np.float64)
            b = array[:, :, 2].astype(np.float64)
            gray = (0.299 * r + 0.587 * g + 0.114 * b)
            gray_u8 = np.clip(gray.round(), 0, 255).astype(np.uint8)
        else:
            h, w = array.shape
            gray_u8 = array.astype(np.uint8)

        num_vertices = int(h * w)
        intens = gray_u8.reshape(-1).astype(np.float64)

        edges = []
        weights = []
        sigI = float(max(1e-12, sigma_I))

        def vid(y: int, x: int) -> int:
            return y * w + x

        for y in range(h):
            for x in range(w):
                i = vid(y, x)
                if x + 1 < w:
                    j = vid(y, x + 1)
                    d = abs(intens[i] - intens[j])
                    wgt = float(np.exp(- (d * d) / (sigI * sigI)))
                    if wgt > 0.0:
                        edges.append((i, j))
                        weights.append(wgt)
                if y + 1 < h:
                    j = vid(y + 1, x)
                    d = abs(intens[i] - intens[j])
                    wgt = float(np.exp(- (d * d) / (sigI * sigI)))
                    if wgt > 0.0:
                        edges.append((i, j))
                        weights.append(wgt)

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

    # superpixel mode: compute mean intensities and 4-neighbor style weights on adjacency
    image_u8 = ensure_rgb_uint8(array, is_rgb)
    segments = compute_slico_segments(
        array, is_rgb,
        n_segments=int(n_segments), compactness=float(compactness), sigma=float(sigma), start_label=int(start_label)
    )
    mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
    edges = compute_segment_adjacency_edges(segments)
    num_nodes = int(mean_r.size)
    sigI = float(max(1e-12, sigma_I))
    weights: list[float] = []
    for u, v in edges:
        d = abs(int(mean_gray[u]) - int(mean_gray[v]))
        wgt = float(np.exp(- (d * d) / (sigI * sigI)))
        weights.append(wgt)

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


register_graph_builder("prob4", build_prob4_pixel_graph)


