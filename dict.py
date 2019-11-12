import networkx as nx
import json
from graph import 
# import jsonpickle


class ModelObject:
    def __init__(self, data):
        self.id = data['id']

        self.name = data['name']

        slices = data['slices']
        slices = list(sorted(slices, key=lambda slice: slice[0]))
        assert slices[0][0] == 0.
        assert slices[-1][0] == 1.
        assert all(map(lambda s: s[0] >= 0. and s <= 1., slices))
        slices = list(map(lambda s: s if len(s) == 2 else s + [0], slices))
        assert all(map(lambda s: len(s) == 3, slices))
        self.slices = slices

        self.length = data['length']
        if not isinstance(self.length, list):
            self.length = [self.length, self.length]
        assert len(self.length) == 2

    def check_if_applies(widths):
        def check(values):
            length = len(values)
            for start_desc, end_desc in zip(self.slices[:-1], self.slices[1:]):
                start, start_width, start_dev = start_desc
                end, end_width, end_dev = start_desc
                start, end = int(start * length), int(end * length)
                low = min(start_width - start_dev, end_width - end_dev)
                high = min(start_width + start_dev, end_width + end_dev)
                if not all(lambda val: val >= low and val <= high):
                    return False
            return True
        
        for length in range(self.length[1], self.length[0] - 1, -1):
            if len(widths) >= length and check(widths[:length]):
                return length
        return -1
        
def find_objects_on_branch(model_objects, branch):
    start = 0
    objects = []
    while start != len(branch):
        objects_lenghts = []
        for model_object in model_objects:
            objects_lenghts.append(model_object.check_if_applies(model_objects[start:]))
        max_length = max(objects_lenghts)
        if max_length > 0:
            index = objects_lenghts.index(max_length):
            objects.append(model_objects[index].id, start, start + max_length)
            start += max_length
        else:
            start += 1:


def find_objects_on_graph(model_objects, graph):
    branches, joints = divide_graph_by_branches(graph)
    found_objects = []
    for branch in branches:
        branch_widths = list(map(lambda node: node.width, branch))
        found_objects_on_branch = find_objects_on_branch(model_objects, branch_widths)
        for found_object in found_objects_on_branch:
            object_id, start, end = found_object
            found_objects.append((objects, branch[start], branch[end]))
    return found_object


class FoundObject:
    def __init__(self, model_id, coord=None, angle=None):
        self.model_id = model_id
        self.coord = coord
        self.angle = angle


def load(filename):
    with open(filename) as f:
        return json.loads(f.read())
        # return jsonpickle.decode(f.read())


def save(found_objects_graph, filename='found_objects'):
    data = nx.readwrite.json_graph.adjacency_data(found_objects_graph)
    s = json.dumps(data)
    with open(filename, 'w') as dest:
        dest.write(s)


def parse_model_objects(input_data):
    result = []
    for d in input_data:
        result.append(ModelObject(d))
    return result


if __name__ == '__main__':
    data = load('sample_input.json')
    model_objects = parse_model_objects(data)
    # print(model_objects)