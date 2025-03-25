import numpy as np
import segimage


def main():
    image_path = '5096.jpg'
    graph_path = f'images/graph/{image_path.replace(".jpg", ".png")}'
    segimage_path = f'images/segmented/{image_path}'
    img, segments, graph = segimage.img2graph(f'images/original/{image_path}')
    partition = graph.community_hedonic_queue()
    segimage.save_graph_as_image(graph, graph_path, partition)
    resolutions = np.linspace(0.01, 1, 11)
    for res in resolutions:
        partition = graph.community_hedonic_queue(resolution=res)
        res_path = segimage_path.replace('.jpg', f'_{res:.2f}.png')
        segimage.save_segmented_image(partition, segments, img, res_path)
    return True


__name__ == '__main__' and main()