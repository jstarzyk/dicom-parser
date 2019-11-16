# import networkx as nx
import json
from graph import *
# import jsonpickle


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
                    end_index=start,
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
            start += max_length + 1
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
            found_objects.append(FoundObject(object_desc))
        branch.found_objects = found_objects
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
        return text


class GraphOfFoundObjects:
    def __init__(self, graph, model_objects, name="Graph"):
        self.start_branch, self.joints = find_objects_on_graph(model_objects, graph)
        self.name = name
        self.length = len(graph.nodes)


    def __repr__(self):
        stack = [(self.start_branch, '\t')]
        text = "%s (length=%d):\n" % (self.name, self.length)
        while stack:
            curr_branch, prefix = stack.pop(0)
            text += prefix + str(curr_branch.found_objects[0]) + '\n'
            for found_object in curr_branch.found_objects[1:]:
                text += prefix[:-4] + '\t' + str(found_object) + '\n'
            if len(curr_branch.next) == 1:
                stack.append((curr_branch.next[0], prefix[:-4] + '   |'))
            else:
                for next_branch in curr_branch.next:
                    stack.append((next_branch, prefix + '∟___'))
        return text


    @staticmethod
    def create_and_write_graphs(file, graphs, model_objects):
        graphs_with_objects = []
        for i, graph in enumerate(graphs):
            graph_with_objects = GraphOfFoundObjects(graph, model_objects, "Graph #%d" % i)
            file.write(graph_with_objects.__repr__())
            graphs_with_objects.append(graph_with_objects)
        return graphs_with_objects


def load_objects(filename):
    with open(filename) as f:
        objects_desc = json.loads(f.read())
        return [ModelObject(desc) for desc in objects_desc]
        


def save(found_objects_graph, filename='found_objects'):
    data = nx.readwrite.json_graph.adjacency_data(found_objects_graph)
    s = json.dumps(data)
    with open(filename, 'w') as dest:
        dest.write(s)


if __name__ == '__main__':
    model_objects = load_objects('sample_input.json')
    for model_object in model_objects:
        print(model_object)