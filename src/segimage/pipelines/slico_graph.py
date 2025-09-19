"""
Pipeline: SLICO → Graph

Runs the SLICO processor to compute superpixels, then builds a superpixel
adjacency graph by reusing the core logic from the pixel graph with
segment-level features. This composition avoids duplicating processor logic
and keeps the pipeline orchestration separate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import register_pipeline
from ..utils import compute_lbp_float_from_rgb_uint8


def _load_rgb_image(input_path: Path) -> np.ndarray:
    img = Image.open(input_path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def slico_graph_run(
    input_path: Path,
    output_path: Path,
    *,
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
    edge_filter: Optional[str] = None,
    edge_similarity: Optional[float] = None,
) -> bool:
    """Compose SLICO and graph steps to build a superpixel adjacency graph.

    Notes:
      - This pipeline mirrors the behavior of the prior graph_slico processor
        but centralizes composition here to encourage reuse.
    """
    try:
        # Import here to keep optional dependency
        from skimage.segmentation import slic
        from skimage.util import img_as_float
        from igraph import Graph
        from PIL import ImageDraw

        input_path = Path(input_path)
        image_u8 = _load_rgb_image(input_path) if input_path.suffix.lower() != ".npy" else np.load(str(input_path)).astype(np.uint8)
        if image_u8.ndim == 2:
            image_u8 = np.stack([image_u8] * 3, axis=-1).astype(np.uint8)
        if image_u8.ndim == 3 and image_u8.shape[-1] >= 3:
            image_u8 = image_u8[..., :3]

        h, w, _ = image_u8.shape
        image_float = img_as_float(image_u8)

        segments = slic(
            image_float,
            n_segments=int(max(1, n_segments)),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=int(start_label),
            channel_axis=-1,
            slic_zero=True,
        )

        unique_labels, inverse = np.unique(segments, return_inverse=True)
        seg_ids = inverse.reshape(h, w).astype(np.int32)
        num_nodes = int(unique_labels.size)

        # Features per superpixel
        r = image_u8[:, :, 0].astype(np.float64)
        g = image_u8[:, :, 1].astype(np.float64)
        b = image_u8[:, :, 2].astype(np.float64)
        gray_u8 = np.clip((0.299 * r + 0.587 * g + 0.114 * b).round(), 0, 255).astype(np.uint8)

        flat_ids = seg_ids.reshape(-1)
        counts = np.bincount(flat_ids, minlength=num_nodes).astype(np.float64)
        counts[counts == 0] = 1.0

        mean_gray = (np.bincount(flat_ids, weights=gray_u8.reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
        mean_r = (np.bincount(flat_ids, weights=image_u8[:, :, 0].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
        mean_g = (np.bincount(flat_ids, weights=image_u8[:, :, 1].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
        mean_b = (np.bincount(flat_ids, weights=image_u8[:, :, 2].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)

        # LBP per node if needed
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

        # Adjacency between touching superpixels
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

        # Filter edges if requested
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
                c1 = np.stack([mean_r, mean_g, mean_b], axis=1).astype(np.int16)
                diff = c1[edges[:, 0]] - c1[edges[:, 1]]
                dist = np.sqrt((diff[:, 0].astype(np.float64) ** 2) + (diff[:, 1].astype(np.float64) ** 2) + (diff[:, 2].astype(np.float64) ** 2))
                max_d = 255.0 * np.sqrt(3.0)
                d = dist / max_d
                edges = edges[d <= thr]

        # Image visualization
        suffix = output_path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            ys, xs = np.indices((h, w))
            sum_x = np.bincount(flat_ids, weights=xs.reshape(-1).astype(np.float64), minlength=num_nodes)
            sum_y = np.bincount(flat_ids, weights=ys.reshape(-1).astype(np.float64), minlength=num_nodes)
            counts = np.maximum(counts, 1.0)
            cx = (sum_x / counts).astype(np.float64)
            cy = (sum_y / counts).astype(np.float64)

            img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img, "RGBA")
            for e in edges.tolist():
                i, j = int(e[0]), int(e[1])
                draw.line([(cx[i], cy[i]), (cx[j], cy[j])], fill=(0, 0, 0, 64), width=1)
            radius = 2
            for i in range(num_nodes):
                x = cx[i]; y = cy[i]
                r_v = int(mean_r[i]); g_v = int(mean_g[i]); b_v = int(mean_b[i])
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(r_v, g_v, b_v, 255), outline=(r_v, g_v, b_v, 255), width=1)

            out_mode_img = img
            if suffix in (".jpg", ".jpeg"):
                out_mode_img = img.convert("RGB")
            fmt = (
                "PNG" if suffix == ".png" else "JPEG" if suffix in (".jpg", ".jpeg") else "TIFF" if suffix in (".tif", ".tiff") else None
            )
            out_mode_img.save(output_path, fmt) if fmt else out_mode_img.save(output_path)
            return True

        # Graph output
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

        # Save based on suffix like the pixel-graph
        from ..processors.graph import _save_graph as _save_graph_like_pixel
        _save_graph_like_pixel(g, output_path)
        return True
    except Exception as e:
        print(f"Error running SLICO→Graph pipeline: {e}")
        return False


register_pipeline("slico_graph", slico_graph_run)


