from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import colorsys

try:
    import folder_paths  # type: ignore
    from comfy.utils import ProgressBar  # type: ignore
except Exception:  # pragma: no cover - allow import outside ComfyUI
    folder_paths = None  # type: ignore
    ProgressBar = None  # type: ignore

# Ensure local src/ is importable so we can import segimage without PyPI install
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Avoid importing segimage submodules at import-time to prevent hard failures
# when dependencies (e.g., python-igraph, scikit-image) are not yet installed.


def _image_to_array_and_flag(img: Any) -> Tuple[np.ndarray, bool]:
    if img is None:
        raise ValueError("IMAGE input not connected. Please connect a loader (e.g., Load Image) to the node.")

    # Accept path-like: load via PIL
    try:
        from pathlib import Path
        if isinstance(img, (str, bytes, os.PathLike, Path)):
            p = Path(img)
            if p.exists():
                with Image.open(p) as pil:
                    pil = pil.convert("RGB")
                    arr = np.array(pil).astype(np.uint8)
                    return arr, True
    except Exception:
        pass
    # Support ComfyUI IMAGE dict/tensor, PIL, or numpy
    try:
        from nodes import tensor2pil, pil2tensor  # ComfyUI helpers
    except Exception:
        tensor2pil = None  # type: ignore
        pil2tensor = None  # type: ignore
    if tensor2pil is None:
        try:
            from comfy.utils import tensor2pil as _c_tensor2pil  # type: ignore
            tensor2pil = _c_tensor2pil  # type: ignore
        except Exception:
            pass

    # Direct torch tensor handling without tensor2pil
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(img, torch.Tensor):
        try:
            a = img.detach().cpu().numpy()
            # Handle batch dim
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            # Move channels to last
            if a.ndim == 3 and a.shape[0] in (1, 3, 4):
                a = np.moveaxis(a, 0, -1)
            # Scale to uint8 if float
            if np.issubdtype(a.dtype, np.floating):
                a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
            if a.ndim == 2:
                return a, False
            if a.ndim == 3 and a.shape[2] >= 3:
                return a[:, :, :3], True
        except Exception as e:
            try:
                print(f"[SegImage] Failed direct torch to numpy: {e}")
                print(f"[SegImage] Tensor shape: {img.shape}")
            except Exception:
                pass

    # ComfyUI: {"images": [tensor, ...]}
    if isinstance(img, dict):
        t = None
        if "images" in img and isinstance(img["images"], (list, tuple)) and img["images"]:
            t = img["images"][0]
        elif "image" in img:
            t = img["image"]
        # If still None, try to find the first tensor-like value in the dict
        if t is None:
            for v in img.values():
                try:
                    if isinstance(v, torch.Tensor):
                        t = v
                        break
                except Exception:
                    pass
        if t is not None:
            if tensor2pil is not None:
                try:
                    pil = tensor2pil(t)
                    arr = np.array(pil)
                    if arr.ndim == 2:
                        return arr.astype(np.uint8), False
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        return arr[:, :, :3].astype(np.uint8), True
                except Exception:
                    pass
            # Fallback: try basic tensor → numpy (channels last, 0..1)
            try:
                if isinstance(t, torch.Tensor):
                    a = t.detach().cpu().numpy()
                    if a.ndim == 3 and a.shape[0] in (1, 3, 4):
                        a = np.moveaxis(a, 0, -1)
                    a = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    if a.ndim == 2:
                        return a, False
                    if a.ndim == 3 and a.shape[2] >= 3:
                        return a[:, :, :3], True
            except Exception:
                pass

    # Sometimes IMAGE is passed as a list/tuple with a tensor
    if isinstance(img, (list, tuple)) and img:
        first = img[0]
        try:
            if torch is not None and isinstance(first, torch.Tensor):
                return _image_to_array_and_flag(first)
        except Exception:
            pass

    # Recursive search for tensor/PIL/numpy inside arbitrary containers
    def _find_payload(o: Any, depth: int = 0) -> Optional[Tuple[np.ndarray, bool]]:
        if depth > 5:
            return None
        try:
            if torch is not None and isinstance(o, torch.Tensor):
                return _image_to_array_and_flag(o)
        except Exception:
            pass
        if isinstance(o, Image.Image):
            a = np.array(o)
            if a.ndim == 2:
                return a.astype(np.uint8), False
            if a.ndim == 3 and a.shape[2] >= 3:
                return a[:, :, :3].astype(np.uint8), True
        if isinstance(o, np.ndarray):
            if o.ndim == 2:
                return o.astype(np.uint8), False
            if o.ndim == 3 and o.shape[2] >= 3:
                return o[:, :, :3].astype(np.uint8), True
        # Objects with attributes commonly used by ComfyUI for IMAGE
        for attr in ("image", "images", "tensor", "images_tensor", "samples"):
            try:
                if hasattr(o, attr):
                    v = getattr(o, attr)
                    r = _find_payload(v, depth + 1)
                    if r is not None:
                        return r
            except Exception:
                pass
        if isinstance(o, dict):
            for v in o.values():
                r = _find_payload(v, depth + 1)
                if r is not None:
                    return r
        if isinstance(o, (list, tuple)):
            for v in o:
                r = _find_payload(v, depth + 1)
                if r is not None:
                    return r
        return None

    found = _find_payload(img, 0)
    if found is not None:
        return found

    if isinstance(img, Image.Image):
        arr = np.array(img)
        if arr.ndim == 2:
            return arr.astype(np.uint8), False
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3].astype(np.uint8), True
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return img.astype(np.uint8), False
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, :3].astype(np.uint8), True

    # Final fallback: try ComfyUI's tensor2pil directly on the incoming object
    try:
        try:
            from nodes import tensor2pil as _t2p  # type: ignore
        except Exception:
            from comfy.utils import tensor2pil as _t2p  # type: ignore
        try:
            pil = _t2p(img)  # type: ignore[arg-type]
            arr = np.array(pil)
            if arr.ndim == 2:
                return arr.astype(np.uint8), False
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].astype(np.uint8), True
        except Exception:
            pass
    except Exception:
        pass

    try:
        print(f"[SegImage] Unsupported IMAGE type: {type(img)}")
        if isinstance(img, dict):
            print(f"[SegImage] IMAGE dict keys: {list(img.keys())}")
    except Exception:
        pass
    raise ValueError("Unsupported IMAGE input; expected ComfyUI IMAGE, PIL.Image or numpy ndarray")


def _pil_to_comfy_image(pil_img: Image.Image) -> List[Any]:
    # Prefer Comfy's pil2tensor for exact IMAGE type, fallback to manual HWC float tensor
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    try:
        from nodes import pil2tensor  # type: ignore
        t = pil2tensor(pil_img)
        return [t]
    except Exception:
        pass
    try:
        from comfy.utils import pil2tensor as _c_pil2tensor  # type: ignore
        t = _c_pil2tensor(pil_img)
        return [t]
    except Exception:
        pass
    try:
        import torch  # type: ignore
        arr = np.array(pil_img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        t = torch.from_numpy(arr).contiguous()
        return [t]
    except Exception:
        arr = np.array(pil_img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return [arr]


def _normalize_weights(weights: List[float]) -> List[float]:
    if not weights:
        return []
    w = np.asarray(weights, dtype=np.float64)
    min_w = float(np.min(w))
    max_w = float(np.max(w))
    if max_w <= 0.0:
        return [0.0 for _ in weights]
    if max_w == min_w:
        return [1.0 for _ in weights]
    wn = (w - min_w) / (max_w - min_w)
    return wn.tolist()


def _segments_to_preview_image(image: np.ndarray, segments: np.ndarray) -> Image.Image:
    # Lazy import to avoid loading the whole segimage package at node import time
    from segimage.graphs.segments import ensure_rgb_uint8, compute_segment_means_and_centroids

    is_rgb = image.ndim == 3 and image.shape[2] >= 3
    img_u8 = ensure_rgb_uint8(image, is_rgb)
    mean_r, mean_g, mean_b, _, _, _ = compute_segment_means_and_centroids(img_u8, segments)
    unique_labels, label_indices = np.unique(segments, return_inverse=True)
    flat = label_indices.reshape(-1)
    num_segments = unique_labels.size
    h, w = segments.shape
    out = np.zeros((h * w, 3), dtype=np.uint8)
    for sid in range(num_segments):
        out[flat == sid, 0] = mean_r[sid]
        out[flat == sid, 1] = mean_g[sid]
        out[flat == sid, 2] = mean_b[sid]
    out_img = out.reshape(h, w, 3)
    return Image.fromarray(out_img, mode="RGB")


class SegimageSLICO:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_segments": ("INT", {"default": 280, "min": 1, "max": 5000, "tooltip": "Approximate number of superpixels to generate in the image."}),
                "compactness_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100.0, "tooltip": "Balances color proximity and spatial proximity. Higher values prioritize compact shapes."}),
                "smoothing_sigma": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "tooltip": "Standard deviation for Gaussian smoothing applied before segmentation."}),
                "start_from_one": ("BOOLEAN", {"default": True, "tooltip": "Start superpixel labels from 1 instead of 0."}),
                "slico_mode": ("BOOLEAN", {"default": True, "tooltip": "Use SLICO zero-parameter mode for superpixel generation."}),
            },
        }

    RETURN_TYPES = ("SLICO_LABELS", "IMAGE")
    RETURN_NAMES = ("labels", "preview")
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(
        self,
        image: Any,
        num_segments: int,
        compactness_factor: float,
        smoothing_sigma: float,
        start_from_one: bool,
        slico_mode: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        from segimage.graphs.segments import compute_slico_segments

        try:
            arr, is_rgb = _image_to_array_and_flag(image)
            segments = compute_slico_segments(
                arr,
                is_rgb,
                n_segments=int(num_segments),
                compactness=float(compactness_factor),
                sigma=float(smoothing_sigma),
                start_label=1 if start_from_one else 0,
                slic_zero=bool(slico_mode),
            )
            preview = _segments_to_preview_image(arr, segments)
            return (segments, _pil_to_comfy_image(preview))
        except Exception as e:
            try:
                import traceback
                print("[SegImage: SLICO] Failed to compute superpixels. This often means 'scikit-image' is not installed in the ComfyUI Python environment.")
                print(f"[SegImage: SLICO] Error: {e}")
                traceback.print_exc()
            except Exception:
                pass
            # Graceful fallback for empty/invalid IMAGE to avoid crashing background checks
            fallback_segments = np.zeros((1, 1), dtype=np.int32)
            preview = Image.new("RGB", (1, 1), (0, 0, 0))
            return (fallback_segments, _pil_to_comfy_image(preview))


class SegimageGraphView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "graph": ("SEG_GRAPH",),
            },
            "optional": {
                "node_radius": ("INT", {"default": 2, "min": 1, "max": 20, "tooltip": "Radius size for drawing nodes in the preview."}),
                "show_nodes": ("BOOLEAN", {"default": True, "tooltip": "Whether to display nodes in the graph preview."}),
                "edge_width_max": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Maximum edge width in pixels."}),
                "edge_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "tooltip": "Minimum edge weight threshold to display (0.0 shows all)."}),
                "labels": ("COMMUNITY_LABELS",),
                "palette": (("bw", "rainbow"), {"default": "rainbow"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "run"
    CATEGORY = "SegImage"

    @classmethod
    def _render_preview(cls, g: Any, *, node_radius: int = 2, show_nodes: bool = True, edge_width_max: int = 3, edge_min: float = 0.0, labels: Optional[np.ndarray] = None, palette: Optional[str] = None) -> Image.Image:
        from igraph import Graph
        if not isinstance(g, Graph):
            raise ValueError("graph must be an igraph.Graph")
        communities = None
        if labels is not None:
            communities = labels.tolist()
        elif "community" in g.vs.attributes():
            communities = g.vs["community"]

        comm_to_color = None
        if communities is not None and palette is not None and len(communities) > 0:
            unique_comms = sorted(set(communities))
            num_comms = len(unique_comms)
            if num_comms > 1:
                comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
                comm_to_color = {}
                for c in unique_comms:
                    val = comm_to_idx[c] / (num_comms - 1.0)
                    if palette == "rainbow":
                        red, green, blue = colorsys.hsv_to_rgb(val * 0.8, 1.0, 1.0)
                        color = (int(255 * red), int(255 * green), int(255 * blue))
                    else:
                        gray = int(255 * val)
                        color = (gray, gray, gray)
                    comm_to_color[c] = color

        try:
            w = int(g["width"]); h = int(g["height"])
        except Exception:
            n = int(g.vcount())
            side = int(max(1, np.sqrt(max(1, n))))
            w = h = side

        try:
            node_mode = str(g["node_mode"]).lower()
        except Exception:
            node_mode = "pixel"

        draw_node_r = max(1, int(node_radius))
        step_node_r = 2 if node_mode == "superpixel" else draw_node_r
        step = 3 * (2 * step_node_r)
        half = step // 2
        out_w = max(1, w * step)
        out_h = max(1, h * step)

        base = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 255))
        overlay = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        try:
            weights = list(g.es["weight"])
        except Exception:
            weights = [1.0] * g.ecount()  # Treat unweighted as full weight

        if weights:
            normalized_weights = _normalize_weights(weights)
            alphas = [32.0 + (220.0 - 32.0) * v for v in normalized_weights]
            widths = [1 + int((edge_width_max - 1) * v) for v in normalized_weights]
        else:
            alphas = [64] * g.ecount()
            widths = [1] * g.ecount()

        # Filter edges based on edge_min
        edgelist = []
        filtered_alphas = []
        filtered_widths = []
        for e_idx, (u, v) in enumerate(g.get_edgelist()):
            if normalized_weights[e_idx] < edge_min:
                continue
            edgelist.append((u, v))
            filtered_alphas.append(alphas[e_idx])
            filtered_widths.append(widths[e_idx])

        width = int(w)
        if node_mode == "superpixel" and all(k in g.vs.attributes() for k in ("cx", "cy")):
            cx = np.array(g.vs["cx"], dtype=np.float64)
            cy = np.array(g.vs["cy"], dtype=np.float64)
        else:
            cx = cy = None

        for e_idx, (u, v) in enumerate(edgelist):
            a = int(filtered_alphas[e_idx])
            ew = int(filtered_widths[e_idx])

            if node_mode == "superpixel" and cx is not None and cy is not None:
                ux = float(cx[int(u)]); uy = float(cy[int(u)])
                vx = float(cx[int(v)]); vy = float(cy[int(v)])
                if comm_to_color is not None:
                    comm_u = communities[int(u)]
                    comm_v = communities[int(v)]
                    cu = comm_to_color[comm_u]
                    cv = comm_to_color[comm_v]
                    col = tuple(((np.array(cu) + np.array(cv)) // 2).tolist() + [a])
                elif all(k in g.vs.attributes() for k in ("r", "g", "b")):
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

            uy, ux = divmod(int(u), width)
            vy, vx = divmod(int(v), width)
            if comm_to_color is not None:
                comm_u = communities[int(u)]
                comm_v = communities[int(v)]
                cu = comm_to_color[comm_u]
                cv = comm_to_color[comm_v]
                col = tuple(((np.array(cu) + np.array(cv)) // 2).tolist() + [a])
            elif all(k in g.vs.attributes() for k in ("r", "g", "b")):
                c1 = np.array([g.vs[int(u)]["r"], g.vs[int(u)]["g"], g.vs[int(u)]["b"]], dtype=int)
                c2 = np.array([g.vs[int(v)]["r"], g.vs[int(v)]["g"], g.vs[int(v)]["b"]], dtype=int)
                col = tuple(((c1 + c2) // 2).tolist() + [a])
            else:
                v1 = int(g.vs[int(u)].get("gray", 128)); v2 = int(g.vs[int(v)].get("gray", 128))
                vavg = (v1 + v2) // 2
                col = (vavg, vavg, vavg, a)
            x1 = ux * step + half; y1 = uy * step + half
            x2 = vx * step + half; y2 = vy * step + half
            draw.line([(x1, y1), (x2, y2)], fill=col, width=ew)

        if show_nodes:
            node_alpha = 220
            if node_mode == "superpixel" and cx is not None and cy is not None:
                has_rgb = all(k in g.vs.attributes() for k in ("r", "g", "b"))
                for i in range(g.vcount()):
                    px = int(round(float(cx[i]) * step + half))
                    py = int(round(float(cy[i]) * step + half))
                    tlx = px - draw_node_r
                    tly = py - draw_node_r
                    brx = px + draw_node_r
                    bry = py + draw_node_r
                    if comm_to_color is not None:
                        comm = communities[i]
                        r, g_, b_ = comm_to_color[comm]
                    elif has_rgb:
                        r = int(g.vs[i]["r"]); g_ = int(g.vs[i]["g"]); b_ = int(g.vs[i]["b"])
                    else:
                        v = int(g.vs[i].get("gray", 128)); r = g_ = b_ = v
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
                        node_idx = yy * w + xx
                        if comm_to_color is not None:
                            comm = communities[node_idx]
                            r, g_, b_ = comm_to_color[comm]
                        elif all(k in g.vs.attributes() for k in ("r", "g", "b")):
                            r, g_, b_ = int(g.vs[node_idx]["r"]), int(g.vs[node_idx]["g"]), int(g.vs[node_idx]["b"])
                        else:
                            v = int(g.vs[node_idx].get("gray", 128)); r = g_ = b_ = v
                        draw.ellipse([(tlx, tly), (brx, bry)], fill=(r, g_, b_, node_alpha), outline=(r, g_, b_, node_alpha), width=1)

        return Image.alpha_composite(base.convert("RGBA"), overlay)

    def run(self, graph: Any, node_radius: int = 2, show_nodes: bool = True, edge_width_max: int = 3, edge_min: float = 0.0, labels: Optional[np.ndarray] = None, palette: str = "rainbow") -> Tuple[Dict[str, Any]]:
        from igraph import Graph
        if not isinstance(graph, Graph):
            print(f"[SegImage] Invalid graph type in GraphView run: {type(graph)} {graph}")
            img = Image.new("RGB", (1, 1), (0, 0, 0))
            return (_pil_to_comfy_image(img),)
        try:
            img = SegimageGraphView._render_preview(graph, node_radius=int(node_radius), show_nodes=bool(show_nodes), edge_width_max=int(edge_width_max), edge_min=float(edge_min), labels=labels, palette=palette)
        except Exception as e:
            print(f"[SegImage] Error rendering graph preview: {str(e)}")
            import traceback
            traceback.print_exc()
            img = Image.new("RGB", (1, 1), (0, 0, 0))
        return (_pil_to_comfy_image(img),)


class SegimageHedonicCommunities:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "graph": ("SEG_GRAPH",),
                "max_communities": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Maximum number of communities to detect."}),
                "resolution": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "tooltip": "Resolution parameter controlling community size (higher values lead to smaller communities)."}),
            },
        }

    RETURN_TYPES = ("SEG_GRAPH", "COMMUNITY_LABELS")
    RETURN_NAMES = ("graph", "labels")
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(self, graph: Any, max_communities: int = 2, resolution: float = 1.0) -> Tuple[Any, np.ndarray]:
        from igraph import Graph

        if not isinstance(graph, Graph):
            raise ValueError("graph must be an igraph.Graph")

        membership: Optional[List[int]] = None
        try:
            from hedonic import Game  # type: ignore

            game = Game(graph)
            attempt_kwargs: List[Dict[str, Any]] = []
            base_kwargs = {"resolution": float(resolution), "max_communities": int(max(1, max_communities))}
            try:
                edge_weights = graph.es["weight"]
                attempt_kwargs.append({**base_kwargs, "edge_weights": edge_weights})
            except Exception:
                pass
            attempt_kwargs.append(dict(base_kwargs))
            attempt_kwargs.append({"max_communities": int(max(1, max_communities))})
            attempt_kwargs.append({})
            part = None
            for kw in attempt_kwargs:
                try:
                    part = game.community_hedonic(**kw)
                    break
                except Exception:
                    continue
            if part is not None:
                membership = list(map(int, part.membership))
        except Exception:
            membership = None

        if membership is None:
            try:
                if "weight" in graph.es.attributes():
                    part = graph.community_leiden(weights=graph.es["weight"], resolution_parameter=float(resolution))
                else:
                    part = graph.community_leiden(resolution_parameter=float(resolution))
                membership = list(map(int, part.membership))
            except Exception:
                part = graph.community_leiden()
                membership = list(map(int, part.membership))

        graph.vs["community"] = membership
        labels = np.array(membership, dtype=np.int32)
        return (graph, labels)


class SegimageGridGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "segments": ("SLICO_LABELS", {"tooltip": "Connect SLICO labels for superpixel mode (overrides to superpixel automatically)."}),
                "edge_filter": (("none", "lbp_eq", "lbp", "gray", "rgb"), {"default": "none", "tooltip": "Filter type for edges."}),
                "edge_similarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "tooltip": "Similarity threshold for edge filtering."}),
            },
        }

    RETURN_TYPES = ("SEG_GRAPH",)
    RETURN_NAMES = ("graph",)
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(
        self,
        image: Any,
        segments: Optional[np.ndarray] = None,
        edge_filter: str = "none",
        edge_similarity: float = 0.0,
    ) -> Tuple[Any, List[Any]]:
        try:
            arr, is_rgb = _image_to_array_and_flag(image)
        except Exception:
            from igraph import Graph
            g = Graph(); g.add_vertices(1); g["width"] = 1; g["height"] = 1
            preview = Image.new("RGB", (1, 1), (0, 0, 0))
            return (g, _pil_to_comfy_image(preview))

        import igraph
        from segimage.graphs.grid import build_grid_pixel_graph

        if segments is not None:
            if arr.shape[:2] != segments.shape:
                raise ValueError("Segments shape must match image height and width.")
            node_mode = "superpixel"

            from segimage.graphs.segments import ensure_rgb_uint8, compute_segment_means_and_centroids, compute_segment_adjacency_edges
            image_u8 = ensure_rgb_uint8(arr, is_rgb)
            mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
            edges = compute_segment_adjacency_edges(segments)
            num_nodes = int(mean_r.size)

            filter_kind: Optional[str] = None
            if edge_filter is not None:
                ef = edge_filter.strip().lower()
                if ef in ("none", ""):
                    filter_kind = None
                elif ef in ("lbp_eq", "lbp", "gray", "rgb"):
                    filter_kind = ef

            lbp_node = None
            if filter_kind in ("lbp_eq", "lbp"):
                from segimage.utils import compute_lbp_float_from_rgb_uint8
                lbp_float = compute_lbp_float_from_rgb_uint8(image_u8)
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

            if edges.size > 0 and filter_kind is not None:
                thr = 1.0 - float(edge_similarity)
                if filter_kind == "lbp_eq":
                    mask = lbp_node[edges[:, 0]] == lbp_node[edges[:, 1]]
                    edges = edges[mask]
                elif filter_kind == "lbp":
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

            h, w = segments.shape
            g = igraph.Graph()
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

        else:
            node_mode = "pixel"
            g = build_grid_pixel_graph(
                arr,
                is_rgb,
                node_mode="pixel",
                edge_filter=edge_filter,
                edge_similarity=edge_similarity,
            )

        # Remove preview = SegimageGraphBuilder()._render_preview(g, node_radius=2, show_nodes=True)
        return (g,)

class SegimageAffinityGraph:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "segments": ("SLICO_LABELS",),
                "radius": ("INT", {"default": 5, "min": 1, "max": 64}),
                "sigma_I": ("FLOAT", {"default": 10.0, "min": 0.0001, "max": 255.0}),
                "sigma_X": ("FLOAT", {"default": 8.0, "min": 0.0001, "max": 512.0}),
            },
        }

    RETURN_TYPES = ("SEG_GRAPH",)
    RETURN_NAMES = ("graph",)
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(
        self,
        image: Any,
        segments: Optional[np.ndarray] = None,
        radius: int = 5,
        sigma_I: float = 10.0,
        sigma_X: float = 8.0,
    ) -> Tuple[Any, List[Any]]:
        try:
            arr, is_rgb = _image_to_array_and_flag(image)
        except Exception:
            from igraph import Graph
            g = Graph(); g.add_vertices(1); g["width"] = 1; g["height"] = 1
            preview = Image.new("RGB", (1, 1), (0, 0, 0))
            return (g, _pil_to_comfy_image(preview))

        import igraph
        from segimage.graphs.affinity import build_affinity_pixel_graph

        if segments is not None:
            if arr.shape[:2] != segments.shape:
                raise ValueError("Segments shape must match image height and width.")
            node_mode = "superpixel"

            from segimage.graphs.segments import ensure_rgb_uint8, compute_segment_means_and_centroids, compute_segment_adjacency_edges
            image_u8 = ensure_rgb_uint8(arr, is_rgb)
            mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
            edges = compute_segment_adjacency_edges(segments)
            num_nodes = int(mean_r.size)

            if is_rgb:
                feats = np.stack([mean_r, mean_g, mean_b], axis=-1).astype(np.float64)
            else:
                feats = mean_gray.reshape(-1, 1).astype(np.float64)
            coords = np.stack([cy, cx], axis=-1).astype(np.float64)

            sigI = float(max(1e-12, sigma_I))
            sigX = float(max(1e-12, sigma_X))
            weights: List[float] = []
            for u, v in edges:
                df = feats[u] - feats[v]
                dist_feature_sq = float(np.dot(df, df))
                dc = coords[u] - coords[v]
                dist_spatial = float(np.hypot(dc[0], dc[1]))
                w_I = float(np.exp(-(dist_feature_sq) / (sigI * sigI)))
                w_X = float(np.exp(-(dist_spatial * dist_spatial) / (sigX * sigX)))
                weights.append(float(w_I * w_X))

            h, w = segments.shape
            g = igraph.Graph()
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

        else:
            node_mode = "pixel"
            g = build_affinity_pixel_graph(
                arr,
                is_rgb,
                node_mode="pixel",
                radius=radius,
                sigma_I=sigma_I,
                sigma_X=sigma_X,
            )

        # Remove preview = SegimageGraphBuilder()._render_preview(g, node_radius=2, show_nodes=True)
        return (g,)

class SegimageProb4Graph:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "segments": ("SLICO_LABELS",),
                "sigma_I": ("FLOAT", {"default": 10.0, "min": 0.0001, "max": 255.0}),
            },
        }

    RETURN_TYPES = ("SEG_GRAPH",)
    RETURN_NAMES = ("graph",)
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(
        self,
        image: Any,
        segments: Optional[np.ndarray] = None,
        sigma_I: float = 10.0,
    ) -> Tuple[Any, List[Any]]:
        try:
            arr, is_rgb = _image_to_array_and_flag(image)
        except Exception:
            from igraph import Graph
            g = Graph(); g.add_vertices(1); g["width"] = 1; g["height"] = 1
            preview = Image.new("RGB", (1, 1), (0, 0, 0))
            return (g, _pil_to_comfy_image(preview))

        import igraph
        from segimage.graphs.prob4 import build_prob4_pixel_graph

        if segments is not None:
            if arr.shape[:2] != segments.shape:
                raise ValueError("Segments shape must match image height and width.")
            node_mode = "superpixel"

            from segimage.graphs.segments import ensure_rgb_uint8, compute_segment_means_and_centroids, compute_segment_adjacency_edges
            image_u8 = ensure_rgb_uint8(arr, is_rgb)
            mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
            edges = compute_segment_adjacency_edges(segments)
            num_nodes = int(mean_r.size)

            sigI = float(max(1e-12, sigma_I))
            weights: List[float] = []
            for u, v in edges:
                d = abs(int(mean_gray[u]) - int(mean_gray[v]))
                wgt = float(np.exp(-(d * d) / (sigI * sigI)))
                weights.append(wgt)

            h, w = segments.shape
            g = igraph.Graph()
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

        else:
            node_mode = "pixel"
            g = build_prob4_pixel_graph(
                arr,
                is_rgb,
                node_mode="pixel",
                sigma_I=sigma_I,
            )

        # Remove preview = SegimageGraphBuilder()._render_preview(g, node_radius=2, show_nodes=True)
        return (g,)

class SegimageContrast4Graph:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "segments": ("SLICO_LABELS",),
                "alpha": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0}),
            },
        }

    RETURN_TYPES = ("SEG_GRAPH",)
    RETURN_NAMES = ("graph",)
    FUNCTION = "run"
    CATEGORY = "SegImage"

    def run(
        self,
        image: Any,
        segments: Optional[np.ndarray] = None,
        alpha: float = 10.0,
    ) -> Tuple[Any, List[Any]]:
        try:
            arr, is_rgb = _image_to_array_and_flag(image)
        except Exception:
            from igraph import Graph
            g = Graph(); g.add_vertices(1); g["width"] = 1; g["height"] = 1
            preview = Image.new("RGB", (1, 1), (0, 0, 0))
            return (g, _pil_to_comfy_image(preview))

        import igraph
        from segimage.graphs.contrast4 import build_contrast4_pixel_graph

        if segments is not None:
            if arr.shape[:2] != segments.shape:
                raise ValueError("Segments shape must match image height and width.")
            node_mode = "superpixel"

            from segimage.graphs.segments import ensure_rgb_uint8, compute_segment_means_and_centroids, compute_segment_adjacency_edges
            image_u8 = ensure_rgb_uint8(arr, is_rgb)
            mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
            edges = compute_segment_adjacency_edges(segments)
            num_nodes = int(mean_r.size)

            h_s, w_s = segments.shape
            feats = np.column_stack([
                mean_gray.astype(np.float64) / 255.0,
                cy.astype(np.float64) / max(1.0, float(h_s)),
                cx.astype(np.float64) / max(1.0, float(w_s)),
            ])
            a = float(max(0.0, alpha))
            weights = []
            dim_scale = float(np.sqrt(max(1, feats.shape[1])))
            for u, v in edges:
                d = float(np.linalg.norm(feats[u] - feats[v]) / dim_scale)
                weights.append(float(np.exp(-a * d)))

            g = igraph.Graph()
            g.add_vertices(int(num_nodes))
            g["width"] = int(w_s)
            g["height"] = int(h_s)
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

        else:
            node_mode = "pixel"
            g = build_contrast4_pixel_graph(
                arr,
                is_rgb,
                node_mode="pixel",
                alpha=alpha,
            )

        # Remove preview = SegimageGraphBuilder()._render_preview(g, node_radius=2, show_nodes=True)
        return (g,)

NODE_CLASS_MAPPINGS = {
    "SegimageSLICO": SegimageSLICO,
    "SegimageGraphView": SegimageGraphView,
    "SegimageHedonicCommunities": SegimageHedonicCommunities,
    "SegimageGridGraph": SegimageGridGraph,
    "SegimageAffinityGraph": SegimageAffinityGraph,
    "SegimageProb4Graph": SegimageProb4Graph,
    "SegimageContrast4Graph": SegimageContrast4Graph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegimageSLICO": "SegImage SLICO Superpixels",
    "SegimageGraphView": "SegImage Graph Preview",
    "SegimageHedonicCommunities": "SegImage Hedonic Communities",
    "SegimageGridGraph": "SegImage Grid Graph",
    "SegimageAffinityGraph": "SegImage Affinity Graph",
    "SegimageProb4Graph": "SegImage Prob4 Graph",
    "SegimageContrast4Graph": "SegImage Contrast4 Graph",
}
