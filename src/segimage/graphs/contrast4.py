from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from igraph import Graph

from .segments import (
    compute_slico_segments,
    compute_segment_means_and_centroids,
    compute_segment_adjacency_edges,
    ensure_rgb_uint8,
)
from . import register_graph_builder

try:
    from skimage.filters import gabor
    from skimage.feature import structure_tensor, structure_tensor_eigvals
except Exception:  # pragma: no cover - graceful fallback if skimage unavailable
    gabor = None  # type: ignore[assignment]
    structure_tensor = None  # type: ignore[assignment]
    structure_tensor_eigvals = None  # type: ignore[assignment]


def _to_grayscale01(array: np.ndarray, is_rgb: bool) -> tuple[np.ndarray, int, int]:
    if is_rgb:
        h, w, _ = array.shape
        r = array[:, :, 0].astype(np.float64)
        g = array[:, :, 1].astype(np.float64)
        b = array[:, :, 2].astype(np.float64)
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
    else:
        h, w = array.shape
        gray = array.astype(np.float64)
    max_v = float(np.max(gray)) if gray.size > 0 else 1.0
    gray01 = gray / 255.0 if max_v > 1.0 else gray
    return gray01, h, w


def _compute_filterbank_features(
    gray01: np.ndarray,
    *,
    gabor_frequencies: Sequence[float],
    gabor_thetas: Sequence[float],
    st_sigmas: Sequence[float],
) -> np.ndarray:
    h, w = gray01.shape
    features: list[np.ndarray] = []

    if gabor is not None and len(tuple(gabor_frequencies)) > 0 and len(tuple(gabor_thetas)) > 0:
        for freq in gabor_frequencies:
            f = float(freq)
            if f <= 0.0:
                continue
            for th in gabor_thetas:
                theta = float(th)
                real, imag = gabor(gray01, frequency=f, theta=theta)
                mag = np.sqrt(real * real + imag * imag)
                features.append(mag)

    if structure_tensor is not None and structure_tensor_eigvals is not None and len(tuple(st_sigmas)) > 0:
        for sg in st_sigmas:
            sigma = float(sg)
            if sigma <= 0.0:
                continue
            Axx, Axy, Ayy = structure_tensor(gray01, sigma=sigma)
            l1, l2 = structure_tensor_eigvals(Axx, Axy, Ayy)
            # Orientation (period pi) and anisotropy
            # phi in [-pi/2, pi/2]
            phi = 0.5 * np.arctan2(2.0 * Axy, (Axx - Ayy + 1e-12))
            cos2 = np.cos(2.0 * phi)
            sin2 = np.sin(2.0 * phi)
            trace = l1 + l2
            # Normalized anisotropy in [0,1)
            ani = (l1 - l2) / (trace + 1e-12)
            features.extend([trace, ani, cos2, sin2])

    if not features:
        # Fallback: just gray intensity
        return gray01.reshape(h * w, 1)

    F = np.stack(features, axis=-1)  # (h, w, C)
    return F.reshape(h * w, F.shape[-1])


def _standardize_features(feats: np.ndarray) -> np.ndarray:
    # Zero-mean, unit-variance per feature for balanced distances
    mu = np.mean(feats, axis=0, keepdims=True)
    sigma = np.std(feats, axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (feats - mu) / sigma


def build_contrast4_pixel_graph(
    array: np.ndarray,
    is_rgb: bool,
    *,
    node_mode: str = "pixel",
    alpha: float = 10.0,
    gabor_frequencies: Optional[Sequence[float]] = (0.05, 0.10, 0.20),
    gabor_thetas: Optional[Sequence[float]] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0),
    st_sigmas: Optional[Sequence[float]] = (1.0, 2.0),
    standardize: bool = True,
    # superpixel (SLICO) parameters
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> Graph:
    """
    Paper-faithful contrast-affinity 4-neighborhood graph.

    Approximates Galun et al.'s multiscale aggregation cues by building a
    per-pixel feature vector that includes multiscale, multi-orientation
    Gabor magnitudes and multi-scale shape elements derived from the
    structure tensor (energy, anisotropy, orientation encoding). Edge
    weights connect 4-neighbor pixels with an affinity computed from the
    standardized Euclidean feature distance.

    Parameters
    - array, is_rgb: input image and format flag
    - alpha: positive scale controlling sharpness; higher => sparser edges
    - gabor_frequencies: cycles-per-pixel for Gabor filters
    - gabor_thetas: orientations in radians
    - st_sigmas: structure tensor scales (in pixels)
    - standardize: if True, standardize each feature dimension
    """
    mode = (node_mode or "pixel").strip().lower()
    gray01, h, w = _to_grayscale01(array, is_rgb)

    if mode == "pixel":
        gf = np.ascontiguousarray(gray01, dtype=np.float64)
        freqs = tuple(gabor_frequencies or ())
        thetas = tuple(gabor_thetas or ())
        sigmas = tuple(st_sigmas or ())

        feats = _compute_filterbank_features(gf, gabor_frequencies=freqs, gabor_thetas=thetas, st_sigmas=sigmas)
        if standardize and feats.shape[1] > 0:
            feats = _standardize_features(feats)

        # Build 4-neighborhood grid graph
        num_vertices = int(h * w)
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        a = float(max(0.0, alpha))

        def vid(y: int, x: int) -> int:
            return y * w + x

        # Precompute to avoid repeated sqrt size factor explosion with high-D
        dim_scale = float(np.sqrt(max(1, feats.shape[1])))

        for y in range(h):
            base = y * w
            for x in range(w):
                i = base + x
                fi = feats[i]
                if x + 1 < w:
                    j = i + 1
                    d = float(np.linalg.norm(fi - feats[j]) / dim_scale)
                    wgt = float(np.exp(-a * d))
                    if wgt > 0.0:
                        edges.append((i, j))
                        weights.append(wgt)
                if y + 1 < h:
                    j = i + w
                    d = float(np.linalg.norm(fi - feats[j]) / dim_scale)
                    wgt = float(np.exp(-a * d))
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
            # store original gray (uint8 if present) or float scaled to 0..255
            if array.dtype == np.uint8:
                g.vs["gray"] = array.reshape(-1).astype(int).tolist()
            else:
                g.vs["gray"] = np.clip((gray01 * 255.0 + 0.5).astype(np.int64), 0, 255).reshape(-1).tolist()
        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights
        return g

    # superpixel mode: compute features on mean-intensity image and connect adjacency with contrast weights
    image_u8 = ensure_rgb_uint8(array, is_rgb)
    segments = compute_slico_segments(
        array, is_rgb,
        n_segments=int(n_segments), compactness=float(compactness), sigma=float(sigma), start_label=int(start_label)
    )
    mean_r, mean_g, mean_b, mean_gray, cx, cy = compute_segment_means_and_centroids(image_u8, segments)
    edges = compute_segment_adjacency_edges(segments)
    num_nodes = int(mean_r.size)

    # Build a simple feature vector per superpixel: use [mean_gray, cx_norm, cy_norm]
    h_s, w_s = segments.shape
    feats = np.column_stack([
        mean_gray.astype(np.float64) / 255.0,
        cy.astype(np.float64) / max(1.0, float(h_s)),
        cx.astype(np.float64) / max(1.0, float(w_s)),
    ])
    if standardize and feats.shape[1] > 0:
        feats = _standardize_features(feats)

    a = float(max(0.0, alpha))
    weights: list[float] = []
    dim_scale = float(np.sqrt(max(1, feats.shape[1])))
    for u, v in edges:
        d = float(np.linalg.norm(feats[u] - feats[v]) / dim_scale)
        wgt = float(np.exp(-a * d))
        weights.append(wgt)

    g = Graph()
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
    return g


register_graph_builder("contrast4", build_contrast4_pixel_graph)

