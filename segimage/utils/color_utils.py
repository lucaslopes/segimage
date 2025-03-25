import random
import numpy as np

def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    hex = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return hex

def get_partition_colors(partition, seed, as_rgb=False):
    """Return a list of colors for each community ID in the partition."""
    membership = partition.membership
    unique_communities = sorted(set(membership))
    random.seed(seed)
    color_list = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in unique_communities
    ]
    color_list_rgb = [hex_to_rgb(color) for color in color_list] if as_rgb else color_list
    community_to_color = {
        c: color_list_rgb[i] for i, c in enumerate(unique_communities)
    }
    return [community_to_color[m] for m in membership]
