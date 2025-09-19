from __future__ import annotations

from typing import Optional

import numpy as np
from igraph import Graph

from ..utils import compute_lbp_float_from_rgb_uint8
from .segments import (
    compute_slico_segments,
    compute_segment_means_and_centroids,
    compute_segment_adjacency_edges,
    ensure_rgb_uint8,
)
from ..processors.graph import _build_8_neighbor_edges
from . import register_graph_builder


def build_grid_pixel_graph(
    array: np.ndarray,
    is_rgb: bool,
    *,
    node_mode: str = "pixel",
    edge_filter: Optional[str] = None,
    edge_similarity: float = 0.0,
    # superpixel (SLICO) parameters
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> Graph:
    mode = (node_mode or "pixel").strip().lower()
    if mode not in ("pixel", "superpixel"):
        raise ValueError("node_mode must be 'pixel' or 'superpixel'")

    filter_kind: Optional[str] = None
    if edge_filter is not None:
        ef = edge_filter.strip().lower()
        if ef in ("none", ""):
            filter_kind = None
        elif ef in ("lbp_eq", "lbp", "gray", "rgb"):
            filter_kind = ef
        else:
            raise ValueError(f"Unsupported edge_filter: {edge_filter}")
    similarity_value: float = float(edge_similarity)
    if similarity_value < 0.0 or similarity_value > 1.0:
        raise ValueError("edge_similarity must be within [0, 1]")

    if mode == "pixel":
        if is_rgb:
            h, w, _ = array.shape
        else:
            h, w = array.shape
        num_vertices = int(h * w)

        lbp_code_flat: Optional[np.ndarray] = None
        gray_u8_flat: Optional[np.ndarray] = None
        rgb_u8_flat: Optional[np.ndarray] = None

        if filter_kind in ("lbp_eq", "lbp"):
            rgb_img = array if is_rgb else np.stack([array] * 3, axis=-1)
            lbp_float = compute_lbp_float_from_rgb_uint8(rgb_img)
            lbp_code_flat = (lbp_float.reshape(-1) * 255.0 + 0.5).astype(np.uint8)
        if filter_kind == "gray":
            if is_rgb:
                r = array[:, :, 0].astype(np.float64)
                g = array[:, :, 1].astype(np.float64)
                b = array[:, :, 2].astype(np.float64)
                gray = (0.299 * r + 0.587 * g + 0.114 * b)
                gray_u8_flat = np.clip(gray.round(), 0, 255).astype(np.uint8).reshape(-1)
            else:
                gray_u8_flat = array.astype(np.uint8).reshape(-1)
        if filter_kind == "rgb":
            if is_rgb:
                rgb = array.astype(np.uint8)
            else:
                rgb = np.stack([array] * 3, axis=-1).astype(np.uint8)
            rgb_u8_flat = rgb.reshape(-1, 3)

        edges = _build_8_neighbor_edges(h, w)
        if edges.size > 0 and filter_kind is not None:
            thr = 1.0 - similarity_value
            if filter_kind == "lbp_eq":
                assert lbp_code_flat is not None
                mask = lbp_code_flat[edges[:, 0]] == lbp_code_flat[edges[:, 1]]
                edges = edges[mask]
            elif filter_kind == "lbp":
                assert lbp_code_flat is not None
                d = np.abs(lbp_code_flat[edges[:, 0]].astype(np.int16) - lbp_code_flat[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
                edges = edges[d <= thr]
            elif filter_kind == "gray":
                assert gray_u8_flat is not None
                d = np.abs(gray_u8_flat[edges[:, 0]].astype(np.int16) - gray_u8_flat[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
                edges = edges[d <= thr]
            elif filter_kind == "rgb":
                assert rgb_u8_flat is not None
                c1 = rgb_u8_flat[edges[:, 0]].astype(np.int16)
                c2 = rgb_u8_flat[edges[:, 1]].astype(np.int16)
                diff = c1 - c2
                dist = np.sqrt((diff[:, 0].astype(np.float64) ** 2) + (diff[:, 1].astype(np.float64) ** 2) + (diff[:, 2].astype(np.float64) ** 2))
                max_d = 255.0 * np.sqrt(3.0)
                d = dist / max_d
                edges = edges[d <= thr]

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
        if edges.size > 0:
            g.add_edges(edges.tolist())
        return g

    # superpixel mode
    image_u8 = ensure_rgb_uint8(array, is_rgb)
    segments = compute_slico_segments(
        array, is_rgb,
        n_segments=int(n_segments), compactness=float(compactness), sigma=float(sigma), start_label=int(start_label)
    )
    mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)

    num_nodes = int(mean_r.size)
    edges = compute_segment_adjacency_edges(segments)

    # Optional LBP per superpixel for filter
    lbp_node: Optional[np.ndarray] = None
    if filter_kind in ("lbp_eq", "lbp"):
        rgb_img = image_u8
        lbp_float = compute_lbp_float_from_rgb_uint8(rgb_img)
        lbp_u8 = (lbp_float * 255.0 + 0.5).astype(np.uint8)
        lbp_flat = lbp_u8.reshape(-1)
        flat_ids = segments.reshape(-1)
        lbp_node = np.zeros(num_nodes, dtype=np.uint8)
        for nid in range(num_nodes):
            m = flat_ids == nid
            if not np.any(m):
                continue
            hist = np.bincount(lbp_flat[m], minlength=256)
            lbp_node[nid] = np.uint8(np.argmax(hist))

    # Apply edge filter in superpixel space
    if edges.size > 0 and filter_kind is not None:
        thr = 1.0 - similarity_value
        if filter_kind == "lbp_eq":
            assert lbp_node is not None
            mask = lbp_node[edges[:, 0]] == lbp_node[edges[:, 1]]
            edges = edges[mask]
        elif filter_kind == "lbp":
            assert lbp_node is not None
            d = np.abs(lbp_node[edges[:, 0]].astype(np.int16) - lbp_node[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
            edges = edges[d <= thr]
        elif filter_kind == "gray":
            d = np.abs(mean_gray[edges[:, 0]].astype(np.int16) - mean_gray[edges[:, 1]].astype(np.int16)).astype(np.float64) / 255.0
            edges = edges[d <= thr]
        elif filter_kind == "rgb":
            c1 = np.stack([mean_r, mean_g, mean_b], axis=-1).astype(np.int16)
            diff = c1[edges[:, 0]] - c1[edges[:, 1]]
            dist = np.sqrt((diff[:, 0].astype(np.float64) ** 2) + (diff[:, 1].astype(np.float64) ** 2) + (diff[:, 2].astype(np.float64) ** 2))
            max_d = 255.0 * np.sqrt(3.0)
            d = dist / max_d
            edges = edges[d <= thr]

    # Build graph
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
    return g


register_graph_builder("grid", build_grid_pixel_graph)


