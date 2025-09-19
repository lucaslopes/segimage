from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from skimage.segmentation import slic
    from skimage.util import img_as_float
except Exception:  # pragma: no cover - optional import for environments without skimage
    slic = None  # type: ignore
    img_as_float = None  # type: ignore


def ensure_rgb_uint8(array: np.ndarray, is_rgb: bool) -> np.ndarray:
    if is_rgb:
        if array.dtype == np.uint8 and array.shape[-1] >= 3:
            return array[:, :, :3]
        if array.shape[-1] >= 3:
            arr = array[:, :, :3]
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
                if info and info.bits > 8:
                    arr = (arr.astype(np.float64) / info.max) * 255.0
            return arr.astype(np.uint8)
    # grayscale → stack
    return np.stack([array] * 3, axis=-1).astype(np.uint8)


def compute_slico_segments(
    array: np.ndarray,
    is_rgb: bool,
    *,
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> np.ndarray:
    if slic is None or img_as_float is None:
        raise RuntimeError("scikit-image is required for superpixel (SLICO) mode but is not available")
    image_u8 = ensure_rgb_uint8(array, is_rgb)
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
    return segments.astype(np.int32)


def compute_segment_means_and_centroids(
    image_u8: np.ndarray,
    segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = segments.shape
    seg_ids = segments.astype(np.int32)
    unique_labels, inverse = np.unique(seg_ids, return_inverse=True)
    num_nodes = int(unique_labels.size)

    r = image_u8[:, :, 0].astype(np.float64)
    g = image_u8[:, :, 1].astype(np.float64)
    b = image_u8[:, :, 2].astype(np.float64)
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    gray_u8 = np.clip(gray.round(), 0, 255).astype(np.uint8)

    flat_ids = inverse.reshape(h, w)
    counts = np.bincount(flat_ids.reshape(-1), minlength=num_nodes).astype(np.float64)
    counts[counts == 0] = 1.0

    mean_gray = (np.bincount(flat_ids.reshape(-1), weights=gray_u8.reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_r = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 0].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_g = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 1].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)
    mean_b = (np.bincount(flat_ids.reshape(-1), weights=image_u8[:, :, 2].reshape(-1).astype(np.float64), minlength=num_nodes) / counts).round().astype(np.uint8)

    # Centroids
    ys, xs = np.mgrid[0:h, 0:w]
    cy = (np.bincount(flat_ids.reshape(-1), weights=ys.reshape(-1).astype(np.float64), minlength=num_nodes) / counts)
    cx = (np.bincount(flat_ids.reshape(-1), weights=xs.reshape(-1).astype(np.float64), minlength=num_nodes) / counts)

    return mean_r, mean_g, mean_b, mean_gray, cx.astype(np.float64), cy.astype(np.float64)


def compute_segment_adjacency_edges(segments: np.ndarray) -> np.ndarray:
    seg_ids = segments.astype(np.int32)
    unique_labels, inverse = np.unique(seg_ids, return_inverse=True)
    mapped = inverse.reshape(seg_ids.shape)

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

    add_pairs(mapped[:, :-1], mapped[:, 1:])
    add_pairs(mapped[:-1, :], mapped[1:, :])
    add_pairs(mapped[:-1, :-1], mapped[1:, 1:])
    add_pairs(mapped[:-1, 1:], mapped[1:, :-1])
    if not edges_set:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(list(edges_set), dtype=np.int64)


