import networkx as nx
import json
from graph import *
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
                low_val, high_val = min(values), max(values)
                if low_val >= low and high_val <= high):
                    return False
            return True
        
        for length in range(self.length[1], self.length[0] - 1, -1):
            if len(widths) >= length and check(widths[:length]):
                return length
        return -1
        
def find_objects_on_branch(model_objects, branch):
    start, start_free_space = 0, 0
    objects = []
    while start != len(branch):
        objects_lenghts = []
        for model_object in model_objects:
            objects_lenghts.append(model_object.check_if_applies(model_objects[start:]))
        max_length = max(objects_lenghts)
        if max_length > 0:
            index = objects_lenghts.index(max_length):
            if start != start_free_space:
                objects.append('Free space', start_free_space, start)
            object_desc = dict(
                name=model_objects[index].id,
                start_index=start,
                end_index=start + max_length,
                length=max_length,
                min_width=min(branch[start:start + max_length]),
                max_width=max(branch[start:start + max_length])
            )
            objects.append(object_desc)
            start += max_length + 1
            start_free_space = start
        else:
            start += 1:


def find_objects_on_graph(model_objects, graph):
    start_branch, joints = divide_graph_by_branches(graph)
    found_objects = []
    stack = [start_branch]
    while stack:
        branch = stack.pop()
        branch_widths = list(map(lambda node: node.width, branch.nodes))
        found_objects_on_branch = find_objects_on_branch(model_objects, branch_widths)
        for found_object in found_objects_on_branch:
            object_desc = found_object
            object_desc['start_coords'] = branch.nodes[object_desc['start_index']].coord
            object_desc['end_coords'] = branch.nodes[object_desc['end_index']].coord
            mid_index = object_desc['start_index'] + (object_desc['end_index'] - object_desc['start_index']) // 2
            object_desc['center_coords'] = branch.nodes[mid_index].coord
            found_objects.append(FoundObject(object_desc))
        branch.found_objects = found_objects
        for next_branch in branch.next:
            stack.add(next_branch)
    return start_branch, joints


class FoundObject:
    def __init__(self, object_desc):
        self.description = object_desc
        assert 'name' in object_desc
    
    def __str__(self):
        text = "Object ID: %s" % self.description['name']
        if 'length' in self.description:
            text += " Length: %d" % self.description['length']
        if 'min_width' in self.description:
            text += " Minimal width: %d" % self.description['min_width']
        if 'max_width' in self.description:
            text += " Maximal width: %d" % self.description['max_width']
        if 'center_coords' in self.description:
            text += " Location: %d, %d" % self.description['center_coords']
        return text


class GraphOfFoundObjects:
    def __init__(self, graph, model_objects, name="Graph"):
        self.start_branch, self.joints = find_objects_on_graph(model_objects, graph)


    def __repr__(self):
        stack = [(self.start_branch, '')]
        text = self.name + ":\n"
        while stack:
            curr_branch, prefix = stack.pop(0)
            for found_object in curr_branch.found_objects:
                text += prefix + found_objects + '\n'
            for next_branch in branch.next:
                stack.append((next_branch, prefix + '\t|'))
        return text


    @staticmethod
    def create_and_write_graphs(file, graphs, model_objects):
        graphs_with_objects = []
        for i, graph in enumerate(graphs):
            graph_with_objects = GraphOfFoundObjects(graph, model_objects, "Graph #%d" % i)
            file.write(graph_with_objects.__repr__())
            graphs_with_objects.append(graph_with_objects)
        return graphs_with_objects


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