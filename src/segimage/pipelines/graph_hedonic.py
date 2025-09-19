"""
Pipeline: Pixel Graph → Hedonic Communities → Visualization

Build an 8-neighbor pixel graph (with optional edge filters), run a hedonic
game community detection with a maximum number of communities, then output
either:
 - An image: nodes (pixels) colored by their community using bw/rainbow
 - A graph: vertex attribute `community` added with community labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from igraph import Graph
from hedonic import Game

from . import register_pipeline
from ..utils import colormap_from_unit_scalar
from ..processors.graph import _load_image_as_array, _save_graph
from ..graphs import get_graph_builder, available_graph_builders


def graph_hedonic_run(
    input_path: Path,
    output_path: Path,
    *,
    K: int = 2,
    palette: str = "bw",
    edge_filter: Optional[str] = None,
    edge_similarity: Optional[float] = None,
    resolution: float = 1.0,
    graph_method: str = "grid",
    radius: int = 5,
    sigma_I: float = 10.0,
    sigma_X: float = 8.0,
    alpha: float = 10.0,
) -> bool:
    try:
        # 1) Build pixel graph in-memory (reuse helpers from pixel-graph)
        array, is_rgb = _load_image_as_array(Path(input_path))
        if is_rgb:
            h, w, _ = array.shape
        else:
            h, w = array.shape
        num_vertices = int(h * w)

        method = graph_method.strip().lower()
        builder = get_graph_builder(method)
        if builder is None:
            names = ", ".join(sorted(available_graph_builders().keys()))
            raise ValueError(f"Unknown graph_method '{graph_method}'. Available: {names}")

        if method == "grid":
            g = builder(array, is_rgb, edge_filter=edge_filter, edge_similarity=(0.0 if edge_similarity is None else float(edge_similarity)))
        elif method == "affinity":
            g = builder(array, is_rgb, radius=int(radius), sigma_I=float(sigma_I), sigma_X=float(sigma_X))
        elif method == "prob4":
            g = builder(array, is_rgb, sigma_I=float(sigma_I))
        elif method == "contrast4":
            g = builder(array, is_rgb, alpha=float(alpha))
        else:
            # Fallback to builder with best-effort kwargs
            g = builder(array, is_rgb)
        
        # Edge weights are optional: only some builders set g.es["weight"].
        # Retrieve safely and pass to hedonic 0.0.9 if available.
        edge_weights = None
        try:
            edge_weights = g.es["weight"]
        except Exception:
            edge_weights = None

        # 2) Hedonic community detection

        game = Game(g)
        attempt_kwargs = []
        base_kwargs = {
            "resolution": float(resolution),
            "max_communities": int(max(1, K)),
        }
        if edge_weights is not None:
            attempt_kwargs.append({**base_kwargs, "edge_weights": edge_weights})
        attempt_kwargs.append(dict(base_kwargs))
        attempt_kwargs.append({"max_communities": int(max(1, K))})
        attempt_kwargs.append({})

        partition = None
        for kw in attempt_kwargs:
            try:
                partition = game.community_hedonic(**kw)
                break
            except Exception:
                continue
        if partition is None:
            # Fallback to igraph Leiden
            try:
                if edge_weights is not None:
                    partition = g.community_leiden(weights=edge_weights, resolution_parameter=float(resolution))
                else:
                    partition = g.community_leiden(resolution_parameter=float(resolution))
            except Exception:
                partition = g.community_leiden()
        membership = np.array(partition.membership, dtype=np.int64)
        g.vs["community"] = membership.astype(int).tolist()
        try:
            num_comms = int(np.unique(membership).size)
            print(f"Communities found (hedonic): {num_comms}")
        except Exception:
            pass

        # 3) Output handling
        suffix = Path(output_path).suffix.lower()
        if suffix in (".graphml", ".gml", ".lg", ".lgl", ".edgelist", ".edges", ".txt", ".pickle", ".pkl"):
            _save_graph(g, Path(output_path))
            return True

        # Image output: map communities → [0,1] → palette
        comms = membership
        # Relabel communities to contiguous 0..C-1
        unique, inv = np.unique(comms, return_inverse=True)
        C = max(1, unique.size)
        vals = inv.astype(np.float64) / float(C - 1) if C > 1 else np.zeros_like(inv, dtype=np.float64)
        img_vals = vals.reshape(h, w)
        img_array, mode = colormap_from_unit_scalar(img_vals, "rainbow" if palette == "rainbow" else "bw")

        img = Image.fromarray(img_array, mode=mode)
        fmt = (
            "PNG"
            if suffix == ".png"
            else "JPEG"
            if suffix in (".jpg", ".jpeg")
            else "TIFF"
            if suffix in (".tif", ".tiff")
            else None
        )
        img.save(output_path, fmt) if fmt else img.save(output_path)
        return True
    except Exception as e:
        print(f"Error running PixelGraph→Hedonic pipeline: {e}")
        return False


register_pipeline("graph_hedonic", graph_hedonic_run)


