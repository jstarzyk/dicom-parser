import networkx as nx
import json
from graph import *


class ModelObject:
    def __init__(self, data):
        self.id = data['id']

        self.name = data['name']

        slices = data['slices']
        slices = list(sorted(slices, key=lambda slice: slice[0]))
        assert slices[0][0] == 0., "First slice must have 0 coordinate"
        assert slices[-1][0] == 1., "Last slice must have 1 coordinate"
        assert all(map(lambda s: s[0] >= 0. and s[0] <= 1., slices)), "All coordinates mast be in (0, 1)"
        slices = list(map(lambda s: s if len(s) == 3 else s + [0], slices))
        assert all(map(lambda s: len(s) == 3, slices)), "Slices must have 2 or 3 parameters"
        self.slices = slices

        self.length = data['length']
        if not isinstance(self.length, list):
            self.length = [self.length, self.length]
        assert len(self.length) == 2
        assert self.length[0] <= self.length[1]

    def check_if_applies(self, widths):
        branch_length = len(widths)
        def check(values):
            length = len(values)
            for start_desc, end_desc in zip(self.slices[:-1], self.slices[1:]):
                start, start_width, start_dev = start_desc
                end, end_width, end_dev = end_desc
                start, end = int(start * length), int(end * length)
                low = min(start_width - start_dev, end_width - end_dev)
                high = max(start_width + start_dev, end_width + end_dev)
                low_val, high_val = np.min(values[start:end]), np.max(values[start:end])
                if low_val < low or high_val > high:
                    return False
            return True

        start = self.length[0] if branch_length > self.length[0] else branch_length
        end = branch_length if branch_length < self.length[1] else self.length[1]
        for length in range(end, start - 1, -1):
            if len(widths) >= length and check(widths[:length]):
                return length
        return -1

    def __repr__(self):
        text = ''
        text += 'ID: %s\n' % self.id
        text += 'Name: %s\n' % self.name
        text += 'Slices: %s\n' % str(self.slices)
        text += 'Length: %s\n' % str(self.length)
        return text

def find_objects_on_branch(model_objects, branch):
    start, start_free_space = 0, 0
    objects = []
    while start < len(branch):
        objects_lenghts = []
        for model_object in model_objects:
            objects_lenghts.append(model_object.check_if_applies(branch[start:]))
        max_length = max(objects_lenghts)
        if max_length > 0:
            index = objects_lenghts.index(max_length)
            if start > start_free_space:
                objects.append(dict(
                    name='Free space',
                    start_index=start_free_space,
                    end_index=start - 1,
                    length=start - start_free_space,
                    min_width=min(branch[start:start + max_length]),
                    max_width=max(branch[start:start + max_length])))
            object_desc = dict(
                name=model_objects[index].id,
                start_index=start,
                end_index=start + max_length - 1,
                length=max_length,
                min_width=min(branch[start:start + max_length]),
                max_width=max(branch[start:start + max_length])
            )
            objects.append(object_desc)
            start += max_length
            start_free_space = start
        else:
            start += 1
    return objects


def find_objects_on_graph(model_objects, graph):
    start_branch, joints = divide_graph_by_branches(graph)
    stack = [start_branch]
    while stack:
        branch = stack.pop()
        branch_widths = list(map(lambda node: node.width, branch.nodes))
        found_objects_desc = find_objects_on_branch(model_objects, branch_widths)
        found_objects = []
        for object_desc in found_objects_desc:
            object_desc['start_coords'] = branch.nodes[object_desc['start_index']].coord
            object_desc['end_coords'] = branch.nodes[object_desc['end_index']].coord
            mid_index = object_desc['start_index'] + (object_desc['end_index'] - object_desc['start_index']) // 2
            object_desc['center_coords'] = branch.nodes[mid_index].coord
            object_desc['max_angle'] = get_maxium_angle(branch.nodes[object_desc['start_index']:object_desc['end_index']])
            found_objects.append(FoundObject(object_desc))
        branch.found_objects = found_objects
        branch.max_angle = get_maxium_angle(branch.nodes)
        for next_branch in branch.next:
            stack.append(next_branch)
    return start_branch, joints


class FoundObject:
    def __init__(self, object_desc):
        self.description = object_desc
        assert 'name' in object_desc

    def __str__(self):
        text = "Object ID: %s;" % self.description['name']
        if 'length' in self.description:
            text += " Length: %d;" % self.description['length']
        if 'min_width' in self.description:
            text += " Minimal width: %.2f;" % self.description['min_width']
        if 'max_width' in self.description:
            text += " Maximal width: %.2f;" % self.description['max_width']
        if 'center_coords' in self.description:
            text += " Location: %d, %d;" % self.description['center_coords']
        if 'max_angle' in self.description:
            text += " Max angle: %0.1f°;" % self.description['max_angle']
        return text


class GraphOfFoundObjects:
    def __init__(self, graph, model_objects, name="Graph"):
        self.start_branch, self.joints = find_objects_on_graph(model_objects, graph)
        self.name = name
        self.length = len(graph.nodes)


    def __repr__(self):
        stack = [(self.start_branch, '1', '\t')]
        text = "%s (length = %d):\n" % (self.name, self.length)
        while stack:
            curr_branch, branch_no, prefix = stack.pop(0)
            text += prefix + "Branch %s (length = %d, max angle = %0.1f°):\n" % \
                (branch_no, len(curr_branch.nodes), curr_branch.max_angle)
            for found_object in curr_branch.found_objects:
                text += prefix + str(found_object) + '\n'
            for i, next_branch in enumerate(curr_branch.next):
                stack.append((next_branch, branch_no + '.' + str(i + 1), prefix + '\t'))
        return text

    @staticmethod
    def find_objects_in_graphs(graphs, model_objects):
        return [GraphOfFoundObjects(graph, model_objects, "Graph #%d" % i) for i, graph in enumerate(graphs)]

    @staticmethod
    def create_and_write_graphs(graphs, model_objects, file=None):
        graphs_with_objects = []
        for i, graph in enumerate(graphs):
            graph_with_objects = GraphOfFoundObjects(graph, model_objects, "Graph #%d" % i)
            graphs_with_objects.append(graph_with_objects)
            if not file is None:
                file.write(graph_with_objects.__repr__())
        return graphs_with_objects

    @staticmethod
    def repr_found_object(found_object, branch_nodes):
        desc = found_object.description
        nodes = branch_nodes[desc["start_index"]:desc["end_index"]]
        return {
            "type": desc["name"],
            "length": get_length_in_px(nodes),
            "min_width": desc["min_width"],
            "max_width": desc["max_width"],
            "start_coords": desc["start_coords"],
            "end_coords": desc["end_coords"],
            "center_coords": desc["center_coords"],
            "max_angle": desc["max_angle"],
        }

    @staticmethod
    def parse_networkx_node(branch):
        return {
            "found_objects": [GraphOfFoundObjects.repr_found_object(o, branch.nodes) for o in branch.found_objects],
            "max_angle": branch.max_angle
        }

    @staticmethod
    def parse_networkx_graph(graph):
        dict_of_lists = {}
        nodes = {}
        stack = [(0, graph.start_branch)]

        while stack:
            i, branch = stack.pop(0)
            adjacent_branches = []
            j = i + 1

            for next_branch in branch.next:
                adjacent_branches.append(j)
                stack.append((j, next_branch))
                j += 1

            nodes[i] = GraphOfFoundObjects.parse_networkx_node(branch)
            dict_of_lists[i] = adjacent_branches

        g = nx.from_dict_of_lists(dict_of_lists)
        for i, attr in nodes.items():
            g.add_node(i, **attr)

        return g

    @staticmethod
    def serialize(graphs):
        return json.dumps(graphs, default=lambda x: x.item())

    @staticmethod
    def parse_networkx_graphs(graphs):
        return [GraphOfFoundObjects.parse_networkx_graph(graph) for graph in graphs]

    @staticmethod
    def to_networkx_json_graph_list(graphs):
        return [nx.readwrite.json_graph.node_link_data(graph) for graph in graphs]


def load_objects(filename):
    with open(filename) as f:
        objects_desc = json.loads(f.read())
        return [ModelObject(desc) for desc in objects_desc]


if __name__ == '__main__':
    model_objects = load_objects('sample_input.json')
    for model_object in model_objects:
        print(model_object)
