"""Shared utility functions for segimage."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Literal

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



def write_meta_for_image(output_path: Path) -> bool:
    """Generate a .meta file alongside an image with per-pixel details.

    The metadata includes image-level info and, for each pixel, the x,y
    coordinate (0-based, where x is column and y is row), the RGB color,
    a grayscale level in [0,1], and an LBP value in [0,1]. The grayscale
    level is computed as the standard luma: 0.299*R + 0.587*G + 0.114*B,
    divided by 255. The LBP (Local Binary Pattern) is computed for each
    pixel using its 8-neighborhood (clockwise order) as a standard 8-bit
    LBP code and normalized by 255 to a float in [0,1].

    The file is written as JSON and streamed to avoid building the entire
    structure in memory.
    """
    try:
        img = Image.open(output_path)
        # Ensure RGB for consistent metadata
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            # Convert grayscale to RGB by stacking channels
            img = img.convert("RGB")

        width, height = img.size
        arr = np.array(img, dtype=np.uint8)  # shape: (H, W, 3)

        # Prepare grayscale levels using luma formula
        r = arr[:, :, 0].astype(np.float64)
        g = arr[:, :, 1].astype(np.float64)
        b = arr[:, :, 2].astype(np.float64)
        gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

        # Compute 8-bit LBP code per pixel (normalized to [0,1])
        # Use 0 for borders where neighbors are missing
        gray255 = (gray * 255.0).round().astype(np.uint8)
        center = gray255
        lbp_code = np.zeros_like(center, dtype=np.uint8)

        # Offsets in clockwise order: TL, T, TR, R, BR, B, BL, L
        # Bits: 7, 6, 5, 4, 3, 2, 1, 0
        # TL (-1,-1) → bit 7
        lbp_code[1:, 1:] |= ((gray255[:-1, :-1] >= center[1:, 1:]).astype(np.uint8) << 7)
        # T (-1,0) → bit 6
        lbp_code[1:, :] |= ((gray255[:-1, :] >= center[1:, :]).astype(np.uint8) << 6)
        # TR (-1,+1) → bit 5
        lbp_code[1:, :-1] |= ((gray255[:-1, 1:] >= center[1:, :-1]).astype(np.uint8) << 5)
        # R (0,+1) → bit 4
        lbp_code[:, :-1] |= ((gray255[:, 1:] >= center[:, :-1]).astype(np.uint8) << 4)
        # BR (+1,+1) → bit 3
        lbp_code[:-1, :-1] |= ((gray255[1:, 1:] >= center[:-1, :-1]).astype(np.uint8) << 3)
        # B (+1,0) → bit 2
        lbp_code[:-1, :] |= ((gray255[1:, :] >= center[:-1, :]).astype(np.uint8) << 2)
        # BL (+1,-1) → bit 1
        lbp_code[:-1, 1:] |= ((gray255[1:, :-1] >= center[:-1, 1:]).astype(np.uint8) << 1)
        # L (0,-1) → bit 0
        lbp_code[:, 1:] |= ((gray255[:, :-1] >= center[:, 1:]).astype(np.uint8) << 0)

        lbp_float = (lbp_code.astype(np.float64) / 255.0)

        meta_path = output_path.with_suffix(output_path.suffix + ".meta")

        import json

        with open(meta_path, "w") as f:
            # Write header
            header = {
                "width": int(width),
                "height": int(height),
                "mode": "RGB",
                "description": "Per-pixel metadata generated by segimage",
            }
            f.write("{\n")
            f.write("  \"width\": %d,\n" % header["width"])
            f.write("  \"height\": %d,\n" % header["height"])
            f.write("  \"mode\": \"%s\",\n" % header["mode"])
            f.write("  \"description\": \"%s\",\n" % header["description"].replace("\"", r"\\\""))
            f.write("  \"pixels\": [\n")

            first = True
            # Stream per-pixel objects row-major
            for y in range(height):
                for x in range(width):
                    r_v = int(arr[y, x, 0])
                    g_v = int(arr[y, x, 1])
                    b_v = int(arr[y, x, 2])
                    gray_v = float(gray[y, x])
                    lbp_v = float(lbp_float[y, x])
                    obj = {
                        "x": x,
                        "y": y,
                        "r": r_v,
                        "g": g_v,
                        "b": b_v,
                        "gray": gray_v,
                        "LBP": lbp_v,
                    }
                    if not first:
                        f.write(",\n")
                    first = False
                    f.write("    " + json.dumps(obj))
            f.write("\n  ]\n}")

        return True
    except Exception as e:  # pragma: no cover - surfaced by callers
        print(f"Error writing .meta file: {e}")
        return False



def load_rgb_image(input_path: Path) -> np.ndarray:
    """Load an image file as RGB uint8 ndarray (H, W, 3)."""
    img = Image.open(input_path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def compute_lbp_float_from_rgb_uint8(image_rgb_u8: np.ndarray) -> np.ndarray:
    """Compute normalized 8-neighbor LBP values in [0,1] from an RGB image.

    Uses luma grayscale and returns a float64 array with shape (H, W).
    """
    r = image_rgb_u8[:, :, 0].astype(np.float64)
    g = image_rgb_u8[:, :, 1].astype(np.float64)
    b = image_rgb_u8[:, :, 2].astype(np.float64)
    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    gray255 = (gray * 255.0).round().astype(np.uint8)
    center = gray255
    lbp_code = np.zeros_like(center, dtype=np.uint8)

    # TL, T, TR, R, BR, B, BL, L → bits 7..0
    lbp_code[1:, 1:] |= ((gray255[:-1, :-1] >= center[1:, 1:]).astype(np.uint8) << 7)
    lbp_code[1:, :] |= ((gray255[:-1, :] >= center[1:, :]).astype(np.uint8) << 6)
    lbp_code[1:, :-1] |= ((gray255[:-1, 1:] >= center[1:, :-1]).astype(np.uint8) << 5)
    lbp_code[:, :-1] |= ((gray255[:, 1:] >= center[:, :-1]).astype(np.uint8) << 4)
    lbp_code[:-1, :-1] |= ((gray255[1:, 1:] >= center[:-1, :-1]).astype(np.uint8) << 3)
    lbp_code[:-1, :] |= ((gray255[1:, :] >= center[:-1, :]).astype(np.uint8) << 2)
    lbp_code[:-1, 1:] |= ((gray255[1:, :-1] >= center[:-1, 1:]).astype(np.uint8) << 1)
    lbp_code[:, 1:] |= ((gray255[:, :-1] >= center[:, 1:]).astype(np.uint8) << 0)

    return lbp_code.astype(np.float64) / 255.0


def colormap_from_unit_scalar(values: np.ndarray, palette: Literal["bw", "rainbow"]) -> Tuple[np.ndarray, str]:
    """Map scalar values in [0,1] to an image array and PIL mode.

    - bw: returns 2D uint8 array (H, W) with mode 'L'
    - rainbow: returns 3D uint8 array (H, W, 3) with mode 'RGB'
    """
    vals = np.clip(values, 0.0, 1.0)
    if palette == "bw":
        img_u8 = (vals * 255.0).round().astype(np.uint8)
        return img_u8, "L"
    # rainbow
    try:
        import matplotlib.pyplot as plt  # type: ignore

        cmap = plt.get_cmap("rainbow")
        # cmap returns RGBA in 0..1 → keep RGB
        h, w = vals.shape
        rgb = np.array(cmap(vals)[:, :, :3], dtype=np.float64)
        img_u8 = (rgb * 255.0).round().astype(np.uint8)
        return img_u8, "RGB"
    except Exception:
        # HSV fallback
        import colorsys

        h, w = vals.shape
        out = np.empty((h, w, 3), dtype=np.uint8)
        # Vectorized mapping through iteration over rows for simplicity
        for i in range(h):
            for j in range(w):
                r_f, g_f, b_f = colorsys.hsv_to_rgb(vals[i, j], 1.0, 1.0)
                out[i, j, 0] = int(r_f * 255)
                out[i, j, 1] = int(g_f * 255)
                out[i, j, 2] = int(b_f * 255)
        return out, "RGB"

