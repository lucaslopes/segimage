from skimage.segmentation import slic
from skimage.util import img_as_float


def create_superpixels(image, n_segments=280, compactness=2):
    """Generate superpixels using SLIC."""
    image_float = img_as_float(image)
    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=1,
        channel_axis=-1)
    return segments

