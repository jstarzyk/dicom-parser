#!/usr/bin/python


import argparse

import pydicom

from dictionary import *
from image_process import get_bin_image


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', dest='source', action='store',
                        default='input.DCM', help='path to input dicom file')
    parser.add_argument('--dict', '-d', dest='dict', action='store',
                        default='sample_input.json', help='path to input dictionary description')
    parser.add_argument('--dest', dest='dest', action='store',
                        default='result.txt', help='path to output found objects description')
    parser.add_argument('--img_dest', dest='img_dest', action='store',
                        default='result.png', help='path to output image with objects')  
    return parser


def get_image_from_dicom(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    image = None
    if len(ds.pixel_array.shape) == 2:
        image = ds.pixel_array
    else:
        image = ds.pixel_array[0]
    return np.uint8(image)


def process_image(image):
    dist, bin_image = get_bin_image(image)
    graphs = list(transform_to_graph(bin_image, r=10))

    for graph in graphs:
        fill_gapes(graph)

    result = list(map(get_largest_path_as_graph, graphs))

    for graph in result:
        add_width_to_nodes(graph, dist)
        add_tree_attributes(graph)
        fill_width_for_conected(graph)

    return result


if __name__ == '__main__':
    arg_parser = init_parser()
    args = arg_parser.parse_args()
    original_image = get_image_from_dicom(args.source)
    model_objects = load_objects(args.dict)
    graphs_processed = process_image(original_image)

    with open(args.dest, 'w') as dest:
        graphs_of_objects = GraphOfFoundObjects.find_objects_in_graphs(graphs_processed, model_objects)
        networkx_json_graph_list = GraphOfFoundObjects.to_networkx_json_graph_list(graphs_of_objects)
        dest.write(GraphOfFoundObjects.serialize(networkx_json_graph_list))
        # print(graphs_of_objects)

    # objects_image = print_objects_on_graphs(graphs_of_objects, original_image, fill=False, method='color_per_object')
    # objects_image = get_graph_image(graphs_processed, original_image, width=False)
    # cv2.imwrite(args.img_dest, objects_image)



    # cv2.imshow('Found Objects', graph_image)
    # cv2.imwrite('tmp.png', graph_image)
    # while cv2.getWindowProperty('Found Objects', 0) >= 0:
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()
