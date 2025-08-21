"""
LBP scalar visualization processor.

Maps each pixel to its 8-neighbor Local Binary Pattern value (normalized to
[0,1]) and renders an image using a chosen palette (bw or rainbow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from . import register_processor
from ..utils import load_rgb_image, compute_lbp_float_from_rgb_uint8, colormap_from_unit_scalar


Palette = Literal["bw", "rainbow"]


def lbp_run(input_path: Path, output_path: Path, *, palette: Palette = "bw") -> bool:
    try:
        input_path = Path(input_path)
        if input_path.suffix.lower() == ".npy":
            array = np.load(str(input_path))
            if array.ndim == 2:
                # grayscale → stack to RGB for LBP pipeline
                image_rgb = np.stack([array] * 3, axis=-1).astype(np.uint8)
            elif array.ndim == 3 and array.shape[-1] >= 3:
                image_rgb = array[..., :3].astype(np.uint8)
            else:
                raise ValueError("Unsupported array shape for LBP processor")
        else:
            image_rgb = load_rgb_image(input_path)

        lbp_vals = compute_lbp_float_from_rgb_uint8(image_rgb)

        # Rank-normalize for rainbow palette: map ranks to [0,1]
        if palette == "rainbow":
            flat = lbp_vals.reshape(-1)
            # argsort twice to get rank positions; ties get increasing order but we normalize by N-1
            order = np.argsort(flat, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(order.size)
            denom = max(1, ranks.size - 1)
            lbp_mapped = ranks.reshape(lbp_vals.shape).astype(np.float64) / float(denom)
            img_array, mode = colormap_from_unit_scalar(lbp_mapped, palette)
        else:
            img_array, mode = colormap_from_unit_scalar(lbp_vals, palette)

        img = Image.fromarray(img_array, mode=mode)
        suffix = output_path.suffix.lower()
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
        print(f"Error running LBP: {e}")
        return False


# Register the processor
register_processor("lbp", lbp_run)



