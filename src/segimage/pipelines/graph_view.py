"""
Pipeline: Build Graph → Render Overlay

Constructs a graph using the selected graph builder method and renders an
overlay image showing nodes on the pixel grid and edges drawn between nodes.
Edge opacity/intensity is proportional to (normalized) weight when weights are
available; otherwise edges are drawn lightly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from igraph import Graph

from . import register_pipeline
from ..processors.graph import _load_image_as_array
from ..graphs import get_graph_builder, available_graph_builders


def _normalize_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    w = np.array(weights, dtype=np.float64)
    min_w = float(np.min(w))
    max_w = float(np.max(w))
    if max_w <= 0.0:
        return [0.0 for _ in weights]
    if max_w == min_w:
        return [1.0 for _ in weights]
    wn = (w - min_w) / (max_w - min_w)
    return wn.tolist()


def graph_view_run(
    input_path: Path,
    output_path: Path,
    *,
    graph_method: str = "grid",
    node_mode: str = "pixel",
    edge_filter: Optional[str] = None,
    edge_similarity: Optional[float] = None,
    radius: int = 5,
    sigma_I: float = 10.0,
    sigma_X: float = 8.0,
    alpha: float = 10.0,
    edge_min: float = 0.0,
    node_radius: int = 2,
    show_nodes: bool = True,
    edge_width_max: Optional[int] = None,
    edge_alpha_min: int = 32,
    edge_alpha_max: int = 220,
    background: str = "white",
    # superpixel (SLICO) parameters
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> bool:
    try:
        array, is_rgb = _load_image_as_array(Path(input_path))
        if is_rgb:
            h, w, _ = array.shape
        else:
            h, w = array.shape

        method = graph_method.strip().lower()
        builder = get_graph_builder(method)
        if builder is None:
            names = ", ".join(sorted(available_graph_builders().keys()))
            raise ValueError(f"Unknown graph_method '{graph_method}'. Available: {names}")

        # Build graph with method-specific parameters
        if method == "grid":
            g = builder(
                array,
                is_rgb,
                node_mode=node_mode,
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
                node_mode=node_mode,
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
                node_mode=node_mode,
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
                node_mode=node_mode,
                alpha=float(alpha),
                n_segments=int(n_segments),
                compactness=float(compactness),
                sigma=float(sigma),
                start_label=int(start_label),
            )
        else:
            g = builder(array, is_rgb, node_mode=node_mode)

        # Gather weights if present
        weights: Optional[list[float]]
        try:
            weights = list(g.es["weight"])  # type: ignore[arg-type]
        except Exception:
            weights = None

        # Determine node mode from graph attribute (fallback to requested)
        try:
            node_mode_effective = str(g["node_mode"]).lower()
        except Exception:
            node_mode_effective = str(node_mode).lower()

        # Prepare spaced grid canvas.
        # For superpixel nodes, decouple draw node radius from grid spacing to avoid scaling the whole image.
        if node_mode_effective == "superpixel":
            draw_node_r = max(1, int(node_radius))
            if draw_node_r <= 3:
                draw_node_r = 10  # large by default for superpixel nodes
            step_node_r = 2  # keep grid spacing compact regardless of draw size
        else:
            draw_node_r = max(1, int(node_radius))
            step_node_r = draw_node_r
        step_node_d = 2 * step_node_r
        step = 3 * step_node_d
        half = step // 2

        # Use original image dimensions for canvas size
        out_w = w * step
        out_h = h * step

        # Base background
        if background == "none":
            base = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        else:
            base = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
        overlay = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Edge drawing: iterate over edges; width and alpha scaled by weight
        alphas: Optional[list[float]] = None
        if weights is not None:
            norm = _normalize_weights(weights)
            # Alpha and width scales
            min_a = int(max(0, min(255, edge_alpha_min)))
            max_a = int(max(0, min(255, edge_alpha_max)))
            if max_a < min_a:
                max_a = min_a
            alphas = [float(min_a + (max_a - min_a) * v) for v in norm]
            max_w_param = int(edge_width_max) if edge_width_max is not None else max(2, 2 * draw_node_r)
            min_w, max_w = 1.0, float(max(1, max_w_param))
            widths = [int(round(min_w + (max_w - min_w) * v)) for v in norm]
        else:
            widths = None
            width_const = int(edge_width_max) if edge_width_max is not None else 1

        # Optional threshold on weight for visibility
        thr = float(edge_min)

        # For each edge
        width = int(w)
        if node_mode_effective == "superpixel":
            cx = np.array(g.vs["cx"], dtype=np.float64) if "cx" in g.vs.attributes() else None
            cy = np.array(g.vs["cy"], dtype=np.float64) if "cy" in g.vs.attributes() else None
        else:
            cx = None
            cy = None

        for e_idx, (u, v) in enumerate(g.get_edgelist()):
            if weights is not None:
                wv = float(weights[e_idx])
                if wv < thr:
                    continue
                a = int(alphas[e_idx]) if alphas is not None else 96
                ew = int(widths[e_idx]) if widths is not None else 1
            else:
                a = 64
                ew = int(widths[e_idx]) if widths is not None else int(width_const)

            if node_mode_effective == "superpixel" and cx is not None and cy is not None:
                ux = float(cx[int(u)]); uy = float(cy[int(u)])
                vx = float(cx[int(v)]); vy = float(cy[int(v)])
                if is_rgb and all(k in g.vs.attributes() for k in ("r", "g", "b")):
                    c1 = np.array([g.vs[int(u)]["r"], g.vs[int(u)]["g"], g.vs[int(u)]["b"]], dtype=int)
                    c2 = np.array([g.vs[int(v)]["r"], g.vs[int(v)]["g"], g.vs[int(v)]["b"]], dtype=int)
                    col = tuple(((c1 + c2) // 2).tolist() + [a])
                else:
                    v1 = int(g.vs[int(u)].get("gray", 128)); v2 = int(g.vs[int(v)].get("gray", 128))
                    vavg = (v1 + v2) // 2
                    col = (vavg, vavg, vavg, a)
                x1 = int(round(ux * step + half)); y1 = int(round(uy * step + half))
                x2 = int(round(vx * step + half)); y2 = int(round(vy * step + half))
                draw.line([(x1, y1), (x2, y2)], fill=col, width=ew)
                continue

            # Pixel mode: map vertex id to grid
            uy, ux = divmod(int(u), width)
            vy, vx = divmod(int(v), width)
            if method in ("prob4", "contrast4"):
                if not ((ux == vx and abs(uy - vy) == 1) or (uy == vy and abs(ux - vx) == 1)):
                    continue
            if is_rgb:
                c1 = array[uy, ux, :3].astype(int)
                c2 = array[vy, vx, :3].astype(int)
                col = tuple(((c1 + c2) // 2).tolist() + [a])
            else:
                v1 = int(array[uy, ux]); v2 = int(array[vy, vx])
                vavg = (v1 + v2) // 2
                col = (vavg, vavg, vavg, a)
            x1 = ux * step + half; y1 = uy * step + half
            x2 = vx * step + half; y2 = vy * step + half
            draw.line([(x1, y1), (x2, y2)], fill=col, width=ew)

        # Draw nodes as circles spaced on the grid
        if show_nodes:
            node_alpha = 220
            if node_mode_effective == "superpixel" and cx is not None and cy is not None:
                has_rgb = all(k in g.vs.attributes() for k in ("r", "g", "b"))
                for i in range(g.vcount()):
                    px = int(round(float(cx[i]) * step + half))
                    py = int(round(float(cy[i]) * step + half))
                    tlx = px - draw_node_r
                    tly = py - draw_node_r
                    brx = px + draw_node_r
                    bry = py + draw_node_r
                    if has_rgb:
                        r = int(g.vs[i]["r"]); g_ = int(g.vs[i]["g"]); b_ = int(g.vs[i]["b"])
                    else:
                        v = int(g.vs[i].get("gray", 128))
                        r = g_ = b_ = v
                    draw.ellipse([(tlx, tly), (brx, bry)], fill=(r, g_, b_, node_alpha), outline=(r, g_, b_, node_alpha), width=1)
            else:
                for yy in range(h):
                    for xx in range(w):
                        cxp = xx * step + half
                        cyp = yy * step + half
                        tlx = cxp - draw_node_r
                        tly = cyp - draw_node_r
                        brx = cxp + draw_node_r
                        bry = cyp + draw_node_r
                        if is_rgb:
                            r, g_, b_ = [int(v) for v in array[yy, xx, :3]]
                        else:
                            v = int(array[yy, xx])
                            r = g_ = b_ = v
                        draw.ellipse([(tlx, tly), (brx, bry)], fill=(r, g_, b_, node_alpha), outline=(r, g_, b_, node_alpha), width=1)

        # Composite overlay onto base
        out_img = Image.alpha_composite(base.convert("RGBA"), overlay)

        # Save
        suffix = Path(output_path).suffix.lower()
        fmt = (
            "PNG" if suffix == ".png" else "JPEG" if suffix in (".jpg", ".jpeg") else "TIFF" if suffix in (".tif", ".tiff") else None
        )
        out_img.save(output_path, fmt) if fmt else out_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error running Graph→View pipeline: {e}")
        return False


register_pipeline("graph_view", graph_view_run)


