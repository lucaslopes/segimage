"""Shared utility functions for segimage."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def normalize_to_uint8(image_data: np.ndarray) -> np.ndarray:
    """Normalize numeric array to uint8 0..255.

    Handles float and integer types and preserves shape.
    """
    arr = image_data
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v == min_v:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr - min_v) / (max_v - min_v)
        return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        min_v = int(np.min(arr))
        max_v = int(np.max(arr))
        if max_v == min_v:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr.astype(np.float64) - min_v) / (max_v - min_v)
        return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Fallback: attempt to cast via float then normalize
    arr = arr.astype(np.float64)
    return normalize_to_uint8(arr)


def save_array_as_image(image_data: np.ndarray, output_path: Path, output_format: str) -> bool:
    """Save a numeric numpy array to an image file with given format.

    For PNG/JPEG, the data is normalized to 0..255 uint8. For TIFF, PIL can
    handle a wider set of types, but we still normalize for consistency.
    """
    try:
        fmt = output_format.lower()
        array_to_save: np.ndarray
        if fmt in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            array_to_save = normalize_to_uint8(image_data)
        else:
            array_to_save = normalize_to_uint8(image_data)

        pil_image = Image.fromarray(array_to_save)
        if fmt == ".png":
            pil_image.save(output_path, "PNG")
        elif fmt in [".jpg", ".jpeg"]:
            pil_image.save(output_path, "JPEG", quality=95)
        elif fmt in [".tif", ".tiff"]:
            pil_image.save(output_path, "TIFF")
        else:
            pil_image.save(output_path)
        return True
    except Exception as e:  # pragma: no cover - surfaced by callers
        print(f"Error saving image: {e}")
        return False


