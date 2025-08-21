"""
Color clustering processor.

Groups image pixels into up to K communities/clusters by frequency of exact
color values. The top K-1 most frequent colors become their own clusters; all
remaining colors are mapped to a single "remain" cluster. Output is a labeled
image where each cluster is assigned a display color according to the selected
palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from PIL import Image

from . import register_processor


Palette = Literal["bw", "rainbow"]


def _ensure_uint8_image(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    # If grayscale float/integer → normalize to 0-255
    if array.ndim == 2:
        arr = array.astype(np.float64)
        min_v = np.min(arr)
        max_v = np.max(arr)
        if max_v == min_v:
            return np.zeros_like(arr, dtype=np.uint8)
        norm = (arr - min_v) / (max_v - min_v)
        return (norm * 255).astype(np.uint8)
    # If multi-channel numeric → clip/scale to 0-255 and cast
    arr = array
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    else:
        # Scale based on dtype range if wider than 8-bit
        info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
        if info and info.bits > 8:
            arr = (arr.astype(np.float64) / info.max) * 255.0
    return arr.astype(np.uint8)


def _load_image(input_path: Path) -> np.ndarray:
    img = Image.open(input_path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _count_colors(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # image: HxW (grayscale) or HxWx3 (RGB)
    if image.ndim == 2:
        flat = image.reshape(-1, 1)
    else:
        flat = image.reshape(-1, image.shape[-1])
    # Use structured array for unique rows (colors)
    if flat.shape[1] == 1:
        colors, counts = np.unique(flat, return_counts=True, axis=0)
    else:
        # View as void to enable unique on rows
        view = flat.view([("c", flat.dtype, flat.shape[1])])
        uniq, counts = np.unique(view, return_counts=True)
        colors = uniq.view(flat.dtype).reshape(-1, flat.shape[1])
    return colors, counts


def _generate_palette(k: int, palette: Palette) -> np.ndarray:
    if palette == "bw":
        if k == 1:
            vals = np.array([0.0], dtype=np.float64)
        else:
            vals = np.linspace(0.0, 1.0, k)
        # grayscale 0..255
        return (vals * 255.0).round().astype(np.uint8)[:, None]
    # rainbow palette → generate RGB using matplotlib if available; else HSV fallback
    try:
        import matplotlib.pyplot as plt  # type: ignore

        cmap = plt.get_cmap("rainbow")
        cols = np.array([cmap(i / max(k, 1))[:3] for i in range(k)], dtype=np.float64)
        return (cols * 255.0).round().astype(np.uint8)
    except Exception:
        # Fallback: evenly spaced HSV → RGB
        import colorsys

        cols = []
        for i in range(k):
            h = i / max(k, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            cols.append([int(r * 255), int(g * 255), int(b * 255)])
        return np.array(cols, dtype=np.uint8)


def _map_to_clusters(image: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    # Count unique colors
    colors, counts = _count_colors(image)
    order = np.argsort(-counts)  # descending by frequency
    colors = colors[order]

    # Take top K-1 as distinct clusters, others merged into remain cluster
    num_distinct = max(0, K - 1)
    top_colors = colors[:num_distinct]

    # Build mapping: color tuple -> cluster_id
    cluster_ids = {}
    for idx, col in enumerate(top_colors):
        key = tuple(col.tolist())
        cluster_ids[key] = idx

    remain_cluster_id = num_distinct  # last cluster index

    # Assign clusters per pixel
    if image.ndim == 2:
        h, w = image.shape
        labels = np.empty((h * w,), dtype=np.int32)
        flat = image.reshape(-1, 1)
    else:
        h, w, _ = image.shape
        labels = np.empty((h * w,), dtype=np.int32)
        flat = image.reshape(-1, image.shape[-1])

    # Vectorized mapping using hash of rows
    # Construct a lookup for top colors
    if flat.shape[1] == 1:
        top_set = set(int(c[0]) for c in top_colors)
        for i, v in enumerate(flat[:, 0]):
            labels[i] = cluster_ids.get((int(v),), remain_cluster_id) if v in top_set else remain_cluster_id
    else:
        top_set = set(tuple(int(x) for x in c) for c in top_colors)
        for i, row in enumerate(flat):
            key = tuple(int(x) for x in row.tolist())
            labels[i] = cluster_ids.get(key, remain_cluster_id) if key in top_set else remain_cluster_id

    return labels.reshape(h, w), top_colors


def color_cluster_run(input_path: Path, output_path: Path, *, K: int = 2, palette: Palette = "bw") -> bool:
    # Load image (accept common formats); also accept .npy as raw arrays
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".npy":
        array = np.load(str(input_path))
        image = _ensure_uint8_image(array)
        if image.ndim == 3 and image.shape[-1] not in (1, 3):
            # Reduce to first 3 channels if more
            image = image[..., :3]
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
    else:
        image = _load_image(input_path)

    if K < 1:
        K = 1

    labels, top_colors = _map_to_clusters(image, K)

    # Build display colors for K clusters
    pal = _generate_palette(K, palette)
    if palette == "bw":
        # pal shape: (K,1) representing grayscale 0..255
        if labels.ndim != 2:
            raise ValueError("Labels must be 2D")
        out = pal[labels][:, :, 0]
        mode = "L"
    else:
        # pal shape: (K,3) RGB
        out = pal[labels]
        mode = "RGB"

    img = Image.fromarray(out.astype(np.uint8), mode=mode)
    # Determine format from suffix
    suffix = output_path.suffix.lower()
    fmt = "PNG" if suffix == ".png" else "JPEG" if suffix in (".jpg", ".jpeg") else "TIFF" if suffix in (".tif", ".tiff") else None
    img.save(output_path, fmt) if fmt else img.save(output_path)
    return True


# Register the processor
register_processor("color_cluster", color_cluster_run)


