"""
Pipeline: SLICO → Superpixel Graph (any method) → Hedonic Communities → Visualization

Segments image with SLICO, builds a superpixel graph using the selected graph
builder (grid/affinity/prob4/contrast4) with node_mode='superpixel', then runs
hedonic game community detection. Outputs either a graph with a community
vertex attribute or an image coloring each pixel by its superpixel's community.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from igraph import Graph
from hedonic import Game
import inspect

from . import register_pipeline
from ..utils import colormap_from_unit_scalar
from ..graphs import get_graph_builder, available_graph_builders
from ..processors.graph import _load_image_as_array, _save_graph
from ..graphs.segments import compute_slico_segments


def slico_graph_hedonic_run(
    input_path: Path,
    output_path: Path,
    *,
    K: int = 2,
    palette: str = "bw",
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
    edge_filter: Optional[str] = None,
    edge_similarity: Optional[float] = None,
    resolution: Optional[float] = None,
    graph_method: str = "grid",
    # optional method-specific params
    radius: int = 5,
    sigma_I: float = 10.0,
    sigma_X: float = 8.0,
    alpha: float = 10.0,
    enforce_max_communities: bool = True,
) -> bool:
    try:
        # Load image
        array, is_rgb = _load_image_as_array(Path(input_path))
        if not is_rgb:
            # SLICO expects RGB; stack grayscale to RGB
            array = np.stack([array] * 3, axis=-1).astype(np.uint8)
            is_rgb = True
        h, w, _ = array.shape

        # Compute SLICO segments (for mapping back to pixels)
        segments = compute_slico_segments(
            array,
            is_rgb,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=int(start_label),
        )
        unique_labels, inverse = np.unique(segments, return_inverse=True)
        seg_ids = inverse.reshape(h, w).astype(np.int32)

        # Build graph via selected builder with node_mode='superpixel'
        method = graph_method.strip().lower()
        builder = get_graph_builder(method)
        if builder is None:
            names = ", ".join(sorted(available_graph_builders().keys()))
            raise ValueError(f"Unknown graph_method '{graph_method}'. Available: {names}")
        if method == "grid":
            g = builder(
                array,
                is_rgb,
                node_mode="superpixel",
                edge_filter=edge_filter,
                edge_similarity=(0.0 if edge_similarity is None else float(edge_similarity)),
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
            )
        elif method == "affinity":
            g = builder(
                array,
                is_rgb,
                node_mode="superpixel",
                radius=int(radius),
                sigma_I=float(sigma_I),
                sigma_X=float(sigma_X),
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
            )
        elif method == "prob4":
            g = builder(
                array,
                is_rgb,
                node_mode="superpixel",
                sigma_I=float(sigma_I),
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
            )
        elif method == "contrast4":
            g = builder(
                array,
                is_rgb,
                node_mode="superpixel",
                alpha=float(alpha),
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
            )
        else:
            g = builder(array, is_rgb, node_mode="superpixel")

        # Hedonic community detection
        game = Game(g)
        # If resolution not provided, derive it from graph edge density
        try:
            density_val = float(g.density())
        except Exception:
            V = max(1, g.vcount())
            E = float(g.ecount())
            density_val = float((2.0 * E) / (float(V) * float(max(1, V - 1))))
        res_value = float(resolution) if resolution is not None else density_val
        kwargs = {
            "resolution": res_value,
            "max_communities": int(max(1, K)),
        }
        # If edges carry weights, pass them when supported (hedonic>=0.0.9)
        try:
            ew = g.es["weight"]
        except Exception:
            ew = None
        if ew is not None:
            kwargs["weights"] = ew
        attempt_kwargs = []
        base_kwargs = {
            "resolution": res_value,
            "max_communities": int(max(1, K)),
        }
        if ew is not None:
            attempt_kwargs.append({**base_kwargs, "weights": ew})
        attempt_kwargs.append(dict(base_kwargs))
        attempt_kwargs.append({"max_communities": int(max(1, K))})
        attempt_kwargs.append({})

        partition = None
        method_used = "hedonic"
        # Introspect supported kwargs for hedonic API and filter dynamically
        try:
            sig = inspect.signature(game.community_hedonic)
            supported = set(sig.parameters.keys())
            # print(f"hedonic.community_hedonic supported params: {sorted(list(supported))}")
        except Exception:
            supported = set()
            print("hedonic.community_hedonic signature not introspectable; will probe args")
        for kw in attempt_kwargs:
            filtered = {k: v for k, v in kw.items() if not supported or k in supported}
            dropped = [k for k in kw.keys() if supported and k not in supported]
            if dropped:
                print(f"Hedonic API does not support: {', '.join(dropped)} (dropping)")
            try:
                partition = game.community_hedonic(**filtered)
                break
            except Exception as e:
                print(f"Error running hedonic: {e}")
                # Probe each keyword individually to find unexpected ones
                try:
                    for k, v in kw.items():
                        try:
                            game.community_hedonic(**{k: v})
                            print(f"Accepted kw: {k}={v}")
                        except Exception as pe:
                            print(f"Rejected kw: {k} -> {pe}")
                except Exception:
                    pass
                continue
        if partition is None:
            try:
                if ew is not None:
                    partition = g.community_leiden(weights=ew, resolution_parameter=float(res_value))
                else:
                    partition = g.community_leiden(resolution_parameter=float(res_value))
                method_used = "leiden"
            except Exception as e:
                print(f"Error running leiden: {e}")
                partition = g.community_leiden()
                method_used = "leiden"
        membership = np.array(partition.membership, dtype=np.int64)
        g.vs["community"] = membership.astype(int).tolist()
        # Optionally enforce K by merging surplus communities into the top-K
        try:
            unique_labels, counts = np.unique(membership, return_counts=True)
            num_comms = int(unique_labels.size)
            if enforce_max_communities and num_comms > max(1, K):
                order = np.argsort(counts)[::-1]
                top_labels = set(unique_labels[order[: int(max(1, K))]].tolist())
                new_membership = membership.copy()
                for v in range(g.vcount()):
                    lbl = int(membership[v])
                    if lbl in top_labels:
                        continue
                    neigh = g.neighbors(v)
                    neigh_labels = [int(membership[n]) for n in neigh if int(membership[n]) in top_labels]
                    if neigh_labels:
                        vals, cnts = np.unique(neigh_labels, return_counts=True)
                        assign = int(vals[int(np.argmax(cnts))])
                    else:
                        assign = int(unique_labels[int(order[0])])
                    new_membership[v] = assign
                membership = new_membership
                g.vs["community"] = membership.astype(int).tolist()
                unique_labels = np.unique(membership)
                num_comms = int(unique_labels.size)
            print(f"Communities found ({method_used}): {num_comms}")
        except Exception:
            pass

        # Output
        suffix = Path(output_path).suffix.lower()
        if suffix in (".graphml", ".gml", ".lg", ".lgl", ".edgelist", ".edges", ".txt", ".pickle", ".pkl"):
            _save_graph(g, Path(output_path))
            return True

        # Image: color each superpixel by its community and expand to full image
        unique, inv = np.unique(membership, return_inverse=True)
        C = max(1, unique.size)
        vals_node = inv.astype(np.float64) / float(C - 1) if C > 1 else np.zeros_like(inv, dtype=np.float64)
        vals_img = vals_node[seg_ids].astype(np.float64)
        img_array, mode = colormap_from_unit_scalar(vals_img, "rainbow" if palette == "rainbow" else "bw")

        out_img = Image.fromarray(img_array, mode=mode)
        fmt = (
            "PNG" if suffix == ".png" else "JPEG" if suffix in (".jpg", ".jpeg") else "TIFF" if suffix in (".tif", ".tiff") else None
        )
        out_img.save(output_path, fmt) if fmt else out_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error running SLICO→Graph→Hedonic pipeline: {e}")
        return False


register_pipeline("slico_graph_hedonic", slico_graph_hedonic_run)


