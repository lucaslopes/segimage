import cv2
import numpy as np
import random

def _hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)

def _get_partition_colors(partition, num_pixels, seed):
    """
    Return a list of colors for each vertex in the partition,
    consistent with the approach in graph_image.py.
    """
    membership = partition.membership
    unique_communities = sorted(set(membership))
    random.seed(seed)
    color_list = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in unique_communities
    ]
    color_list_rgb = [_hex_to_rgb(color) for color in color_list]
    community_to_color = {
        c: color_list_rgb[i] for i, c in enumerate(unique_communities)
    }
    return [community_to_color[m] for m in membership]

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
    colors = _get_partition_colors(partition, label_image.max() + 1, seed=seed)
    visual = np.zeros_like(base_image, dtype=np.uint8)

    for idx, color in enumerate(colors):
        # Print segment index and color
        print(f"Segment index: {idx}, Segment color: {color}")

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
