#!/usr/bin/python


import argparse

import pydicom

from dictionary import *
from image_process import get_bin_image


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', dest='image', action='store',
                        default='input.DCM', help='path to input dicom file')
    parser.add_argument('--dict', '-d', dest='dictionary', action='store',
                        default='sample_input.json', help='path to input dictionary file')
    parser.add_argument('--res', dest='result', action='store',
                        default='graphs.json', help='path to output found objects representation')
    return parser


class ObjectFinder:
    def __init__(self, image_filepath, dictionary_filepath):
        self.dataset = self.load_dicom_dataset(image_filepath)
        self.model_objects = self.load_objects(dictionary_filepath)
        self.mm_per_px = self.get_mm_per_px_ratio(self.dataset)
        self.original_image = self.get_image(self.dataset)
        self.graphs_processed = self.process_image(self.original_image)

    @staticmethod
    def load_dicom_dataset(filepath):
        return pydicom.dcmread(filepath)

    @staticmethod
    def load_objects(filepath):
        with open(filepath) as f:
            objects_desc = json.loads(f.read())
            return [ModelObject(desc) for desc in objects_desc]

    @staticmethod
    def get_mm_per_px_ratio(dataset):
        try:
            pixel_spacing = dataset.PixelSpacing
            ps_x = float(pixel_spacing[0])
            ps_y = float(pixel_spacing[1])
            return ps_x if ps_x == ps_y else None
        except AttributeError:
            return None

    @staticmethod
    def get_image(dataset):
        image = dataset.pixel_array if len(dataset.pixel_array.shape) == 2 else dataset.pixel_array[0]
        return np.uint8(image)

    @staticmethod
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

    def find_objects_on_graphs(self):
        return [GraphOfFoundObjects(graph, self.model_objects, "Graph #%d" % i) for i, graph in
                enumerate(self.graphs_processed)]


if __name__ == '__main__':
    arg_parser = init_parser()
    args = arg_parser.parse_args()

    object_finder = ObjectFinder(args.image, args.dictionary)
    graphs_of_objects = object_finder.find_objects_on_graphs()

    networkx_graphs = GraphOfFoundObjects.parse_networkx_graphs(graphs_of_objects)
    networkx_json_graph_list = GraphOfFoundObjects.to_networkx_json_graph_list(networkx_graphs)
    GraphOfFoundObjects.serialize(networkx_json_graph_list, args.result)
