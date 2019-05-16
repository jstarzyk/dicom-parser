import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy as np
import argparse
from graph import *
from image_process import get_bin_image


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='source', action='store',
                    default='input.DCM', help='path to input dicom file')
    return parser


def get_colors(index, n=3):
    colors = []
    values = np.linspace(0, 255, n)[::-1]
    for r in values:
        for g in values:
            for b in values:
                if r == g and g == b:
                    continue
                colors.append(np.array([r, g, b]))
    return colors[index % len(colors)]

def get_graph_image(graphs, gray=None):
    img = np.full((512, 512, 3), 255)
    if gray is not None:
        for i in np.arange(gray.shape[0]):
            for j in np.arange(gray.shape[1]):
                img[i, j] = [gray[i, j]] * 3
    for i, graph in enumerate(graphs):
        color = get_colors(i)
        for node in graph.nodes:
            y, x = node.coord
            img[y, x] = color
    return img


if __name__ == '__main__':
    arg_parser = init_parser()
    args = arg_parser.parse_args()
    ds = pydicom.dcmread(args.source)
    image = None
    if len(ds.pixel_array.shape) == 2:
        image = ds.pixel_array
    else:
        image = ds.pixel_array[0]
    dist, bin_image = get_bin_image(image)
    graphs = transform_to_graph(bin_image, r=10)
    for graph in graphs:
        fill_gapes(graph)

    graphs_processed = map(get_largest_path_as_graph, graphs)
    graph_image = get_graph_image(graphs_processed, image)
    cv2.imshow('Found Objects', np.uint8( graph_image))

    for graph in graphs_processed:
        add_width_to_nodes(graph, dist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()