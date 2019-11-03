import networkx as nx
import json
# import jsonpickle


class ModelObject:
    def __init__(self, data):
        self.id = data['id']
        self.name = data['name']
        self.slices = data['slices']


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