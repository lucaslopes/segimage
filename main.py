import segimage


def main():
    image_path = '5096.jpg'
    graph = segimage.img2graph(f'images/original/{image_path}')
    segimage.save_graph_as_image(graph, f'images/output/{image_path.replace(".jpg", ".png")}')
    return True


__name__ == '__main__' and main()