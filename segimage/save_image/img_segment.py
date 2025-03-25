import cv2
import numpy as np
from segimage.utils import get_partition_colors

def label_to_image(partition, segments):
    """
    Convert the partition (list of lists with vertex indices) into a 2D label image.
    """
    height, width = segments.shape
    label_image = np.zeros((height, width), dtype=int)
    for cid, vertices in enumerate(partition):
        for v in vertices:
            label_image[segments == (v + 1)] = cid
    return label_image

def color_image_from_labels(label_image, base_image, partition, seed=0):
    """
    Create a color image using the same color mapping as in graph_image.
    """
    h, w = label_image.shape
    colors = get_partition_colors(partition, seed=seed, as_rgb=True)
    visual = np.zeros_like(base_image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            label = label_image[i, j]
            # Use the color from the partition-based mapping
            visual[i, j] = colors[label]

    return visual

def segment_image(partition, segments, original_image):
    label_img = label_to_image(partition, segments)
    visual_img = color_image_from_labels(label_img, original_image, partition)
    return visual_img

def save_segmented_image(partition, segments, original_image, frame_path):
    """
    Save the segmented image to the specified path.
    """
    segmented_image = segment_image(partition, segments, original_image)
    cv2.imwrite(frame_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    print(f"Segmented image saved to {frame_path}")
