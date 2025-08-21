"""
SLICO superpixel processor (SLIC with adaptive compactness).

Generates superpixels using scikit-image's SLIC implementation with
`slic_zero=True` (also known as SLICO). Outputs a color image where each
superpixel region is represented by the average color of the underlying
pixels to provide a visually interpretable segmentation result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import label2rgb

from . import register_processor


def _load_rgb_image(input_path: Path) -> np.ndarray:
    img = Image.open(input_path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _ensure_rgb_uint8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        # Grayscale → stack to RGB
        return np.stack([array] * 3, axis=-1).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] >= 3:
        arr = array[..., :3]
        if arr.dtype == np.uint8:
            return arr
        # Scale/clip to 0..255 then cast
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
            if info and info.bits > 8:
                arr = (arr.astype(np.float64) / info.max) * 255.0
        return arr.astype(np.uint8)
    raise ValueError("Unsupported image array shape for SLICO processor")


def slico_run(
    input_path: Path,
    output_path: Path,
    *,
    n_segments: int = 280,
    compactness: float = 2.0,
    sigma: float = 1.0,
    start_label: int = 1,
) -> bool:
    """Run SLICO superpixel segmentation on an image and save visualization.

    Parameters
    - n_segments: approximate number of superpixels to generate
    - compactness: balance between color proximity and space proximity
    - sigma: width of Gaussian smoothing kernel
    - start_label: starting label index for segments
    """
    try:
        input_path = Path(input_path)
        if input_path.suffix.lower() == ".npy":
            array = np.load(str(input_path))
            image_u8 = _ensure_rgb_uint8(array)
        else:
            image_u8 = _load_rgb_image(input_path)

        # Convert to float in [0,1] for scikit-image
        image_float = img_as_float(image_u8)

        # Determine channel axis for SLIC
        channel_axis = -1  # RGB images

        segments = slic(
            image_float,
            n_segments=int(max(1, n_segments)),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=int(start_label),
            channel_axis=channel_axis,
            slic_zero=True,
        )

        # Produce an interpretable visualization: average color per superpixel
        vis_float = label2rgb(segments, image=image_float, kind="avg")
        vis_u8 = (np.clip(vis_float, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        img = Image.fromarray(vis_u8, mode="RGB")
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
        print(f"Error running SLICO: {e}")
        return False


# Register the processor
register_processor("slico", slico_run)


