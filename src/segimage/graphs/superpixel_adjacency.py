from __future__ import annotations

from typing import Optional

import numpy as np
from igraph import Graph

from ..utils import compute_lbp_float_from_rgb_uint8
from . import register_graph_builder


def build_superpixel_adjacency_graph(
    image_u8: np.ndarray,
    segments: np.ndarray,
    *,
    edge_filter: Optional[str] = None,
    edge_similarity: float = 0.0,
) -> Graph:
    h, w, _ = image_u8.shape
    seg_ids = segments.astype(np.int32)
    unique_labels, inverse = np.unique(seg_ids, return_inverse=True)
    num_nodes = int(unique_labels.size)

    r = image_u8[:, :, 0].astype(np.float64)
    g = image_u8[:, :, 1].astype(np.float64)
    b = image_u8[:, :, 2].astype(np.float64)
    gray_u8 = np.clip((0.299 * r + 0.587 * g + 0.114 * b).round(), 0, 255).astype(np.uint8)

    flat_ids = inverse.reshape(h, w)
    counts = np.bincount(flat_ids.reshape(-1), minlength=num_nodes).astype(np.float64)
    counts[counts == 0] = 1.0

    mean_gray = (np.bincount(flat_ids.reshape(-1), weights=gray_u8.reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_r = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 0].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_g = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 1].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_b = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 2].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)

    # LBP per node (only if required)
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

    lbp_node: Optional[np.ndarray] = None
    if filter_kind in ("lbp_eq", "lbp"):
        lbp_float = compute_lbp_float_from_rgb_uint8(image_u8)
        lbp_u8 = (lbp_float * 255.0 + 0.5).astype(np.uint8)
        lbp_flat = lbp_u8.reshape(-1)
        lbp_node = np.zeros(num_nodes, dtype=np.uint8)
        for nid in range(num_nodes):
            m = flat_ids == nid
            if not np.any(m):
                continue
            hist = np.bincount(lbp_flat[m], minlength=256)
            lbp_node[nid] = np.uint8(np.argmax(hist))

    # Adjacency via touching regions
    edges_set = set()
    def add_pairs(a: np.ndarray, b: np.ndarray):
        mask = a != b
        if not mask.any():
            return
        u = a[mask].reshape(-1)
        v = b[mask].reshape(-1)
        for i in range(u.size):
            s = int(u[i]); t = int(v[i])
            if s == t:
                continue
            if s < t:
                edges_set.add((s, t))
            else:
                edges_set.add((t, s))
    add_pairs(seg_ids[:, :-1], seg_ids[:, 1:])
    add_pairs(seg_ids[:-1, :], seg_ids[1:, :])
    add_pairs(seg_ids[:-1, :-1], seg_ids[1:, 1:])
    add_pairs(seg_ids[:-1, 1:], seg_ids[1:, :-1])
    edges = np.array(list(edges_set), dtype=np.int64) if edges_set else np.empty((0, 2), dtype=np.int64)

    # Build graph
    g = Graph()
    g.add_vertices(num_nodes)
    g["width"] = int(w)
    g["height"] = int(h)
    g["num_segments"] = int(num_nodes)
    g.vs["r"] = mean_r.astype(int).tolist()
    g.vs["g"] = mean_g.astype(int).tolist()
    g.vs["b"] = mean_b.astype(int).tolist()
    g.vs["gray"] = mean_gray.astype(int).tolist()
    if lbp_node is not None:
        g.vs["lbp"] = lbp_node.astype(int).tolist()
    if edges.size > 0:
        g.add_edges(edges.tolist())
    return g


register_graph_builder("superpixel_adjacency", build_superpixel_adjacency_graph)


