import numpy as np
from collections import defaultdict

class Node:

    def __init__(self):
        self.connected = []
        self.visited = False
        self.was_added = False

    def set_coord(self, i, j):
        self.coord = (i, j)


class Graph:

    def __init__(self, nodes=set()):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.add(node)
        node.graph = self

    def union(self, other):
        for node in other.nodes:
            node.graph = self
        self.nodes = self.nodes.union(other.nodes)

    def get_node_by_coord(self, coord):
        for node in self.nodes:
            if node.coord == coord:
                return node


def get_node_matrix(matrix):  # matrix : binary matrix 
    return np.array([np.array([None if elem != 0 else Node() for elem in vector]) for vector in matrix])

def connect_nodes(node_matrix): 
    connected = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for i in np.arange(node_matrix.shape[0]):
        for j in np.arange(node_matrix.shape[1]):
            if node_matrix[i, j] is not None:
                node_matrix[i, j].set_coord(i, j)
                for (d_i, d_j) in connected:
                    next_i, next_j = (i + d_i, j + d_j)
                    if next_i >= 0 and next_i < node_matrix.shape[0] and next_j < node_matrix.shape[1]:
                        if node_matrix[next_i, next_j] is not None:
                            node_matrix[i, j].connected.append(node_matrix[next_i, next_j])
                            node_matrix[next_i, next_j].connected.append(node_matrix[i, j])
    

def get_graphs(node_matrix):
    graphs = []
    for i in np.arange(node_matrix.shape[0]):
        for j in np.arange(node_matrix.shape[1]):
            if node_matrix[i, j] is not None and not node_matrix[i, j].visited:
                graphs.append(construct_graph(node_matrix[i, j]))
    return graphs


def construct_graph(first_node):
    graph = Graph(set())
    stack = [first_node]

    while stack:
        node = stack.pop()
        graph.add_node(node)
        for next_node in node.connected:
            if not next_node.visited:
                next_node.visited = True
                stack.append(next_node)
    return graph
        

def connect_neigh(graphs, node_matrix, r=3):
    for i in np.arange(node_matrix.shape[0] - 1, -1, -1):
        for j in np.arange(node_matrix.shape[1] - 1, -1, -1):
            node = node_matrix[i, j]
            if node is None:
                continue
            for d_i in np.arange(r + 1):
                for d_j in np.arange(r + 1):
                    if d_i > 2 or d_j > 2:
                        next_i, next_j = i + d_i, j + d_j
                        if next_i < node_matrix.shape[0] and next_j < node_matrix.shape[1]:
                            next_node = node_matrix[next_i, next_j]
                            if next_node is not None and node.graph != next_node.graph:
                                node.connected.append(next_node)
                                next_node.connected.append(node)
                                graphs.remove(next_node.graph)                                
                                node.graph.union(next_node.graph)
    for i in np.arange(node_matrix.shape[0]):
        for j in np.arange(node_matrix.shape[1] - 1, -1, -1):
            node = node_matrix[i, j]
            if node is None:
                continue
            for d_i in np.arange(r + 1):
                for d_j in np.arange(r + 1):
                    if d_i > 2 or d_j > 2:
                        next_i, next_j = i - d_i, j + d_j
                        if next_i >= 0 and next_j < node_matrix.shape[1]:
                            next_node = node_matrix[next_i, next_j]
                            if next_node is not None and node.graph != next_node.graph:
                                node.connected.append(next_node)
                                next_node.connected.append(node)
                                graphs.remove(next_node.graph)                                
                                node.graph.union(next_node.graph)


def transform_to_graph(matrix, r=3, remove_round=True, min_size=100):
    node_matrix = get_node_matrix(matrix)
    connect_nodes(node_matrix)
    graphs = get_graphs(node_matrix)
    if (r > 2):
        connect_neigh(graphs, node_matrix, r)
    graphs = filter(lambda graph: len(graph.nodes) > min_size, graphs)
    if remove_round:
        return filter_round(graphs, node_matrix.shape)
    return graphs

def to_csv(graph):
    for index, node in enumerate(graph.nodes):
        node.index = index
    with open('graph_edges.csv', 'w') as dest:
        dest.write('Source,Target\n')
        for node in graph.nodes:
            for next_node in node.connected:
                dest.write('{},{}\n'.format(node.index + 1, next_node.index + 1))


def graph_avg_coords(graph):
    sum_x, sum_y = 0, 0
    num = len(graph.nodes)
    for node in graph.nodes:
        sum_x += node.coord[0]
        sum_y += node.coord[1]
    return (sum_x / num, sum_y / num)


def filter_round(graphs, img_shape, coef=0.9):
    height, width = img_shape
    min_r = min([height, width]) / 2 * coef
    min_r_sqr = min_r ** 2
    get_r = lambda coord: (height / 2 - coord[0]) ** 2 + (width / 2 - coord[1]) ** 2
    def check(graph):
        inside = len(list(filter(lambda node: get_r(node.coord) < min_r_sqr, graph.nodes)))
        return (1.0 * inside / len(graph.nodes)) > 0.9
    return filter(lambda graph: check(graph), graphs)


def Floyd_Warshall(graph):
    size = len(graph.nodes)
    distances = np.full((size, size), np.Inf)
    path_next = np.full((size, size), -1)
    nodes = list(graph.nodes)
    index_map = {node: index for index, node in enumerate(nodes)}
    for i in np.arange(size):
        distances[i, i] = 0
    for node in graph.nodes:
        for next_node in node.connected:
            distances[index_map[node], index_map[next_node]] = 1
            path_next[index_map[node], index_map[next_node]] = index_map[next_node]
    for k in np.arange(size):
        for i in np.arange(size):
            for j in np.arange(size):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
                    path_next[i, j] = k
    return index_map, nodes, distances, path_next

def filter_max_paths(graph):
    index_map, nodes, distances, path_next = Floyd_Warshall(graph)
    (i, j) = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    new_nodes = set()
    for i in np.arange(distances[i, j]):
        new_nodes.add(Node())
    prev = None
    while path_next[i, j] != j:
        new_node = Node()
        new_node.coord = nodes[i].coord
        if prev is not None:
            prev.nodes.connected.append(new_node)
        prev = new_node
        i = j 
        j = path_next[i, j]
        new_nodes.add(new_node)
    return Graph(new_nodes)
    


def Dijkstra(graph, start):
    size = len(graph.nodes)
    distances = {node: np.Inf for node in graph.nodes}
    path = {node: None for node in graph.nodes}
    visited = {node: False for node in graph.nodes}
    distances[start] = 0
    visited[start] = True
    path[start] = start

    ends = []

    stack = [start]
    while stack:
        curr = stack.pop(0)
        visited[curr] = True
        added = False
        for node in curr.connected:
            if not visited[node]:
                added = True
                if distances[curr] + 1 < distances[node]:
                    distances[node] = distances[curr] + 1
                    path[node] = curr
                    stack.append(node)
        if not added:
            ends.append(curr)
    return distances, path, ends


def graph_from_path(graph, path, dest):
    nodes = set()
    prev = Node()
    nodes.add(prev)
    prev.coord = dest.coord
    tmp = dest
    while True:
        tmp = path[tmp]
        new_node = Node()
        nodes.add(new_node)
        prev.connected.append(new_node)
        new_node.connected.append(prev)
        prev = new_node
        new_node.coord = tmp.coord
        try:
            new_node.was_added = tmp.was_added
        except:
            new_node.was_added = False
        if path[tmp] == tmp:
            break
    return Graph(nodes)


def fill_gapes(graph):
    nodes = set(graph.nodes)

    for node in nodes:
        for next_node in node.connected:
            if dist(node.coord, next_node.coord) > 1.5:
                connect(node, next_node, graph)


def connect(node1, node2, graph):
    node1.connected.remove(node2)
    node2.connected.remove(node1)
    if np.abs(node1.coord[0] - node2.coord[0]) > np.abs(node1.coord[1] - node2.coord[1]):
        prev = node1 if node1.coord[0] < node2.coord[0] else node2
        end = node1 if node1.coord[0] > node2.coord[0] else node2
        min_x, min_y = prev.coord
        max_x, max_y = end.coord
        for x in np.arange(min_x + 1, max_x):
            new_node = Node()
            new_node.was_added = True
            y = ((max_y - min_y) * (x - min_x)) / (max_x - min_x) + min_y
            new_node.coord = (int(x), int(y)) 
            new_node.connected.append(prev)
            prev.connected.append(new_node)
            graph.nodes.add(new_node)
            prev = new_node
        end.connected.append(prev)
        prev.connected.append(end)
    else:
        prev = node1 if node1.coord[1] < node2.coord[1] else node2
        end = node1 if node1.coord[1] > node2.coord[1] else node2
        min_x, min_y = prev.coord
        max_x, max_y = end.coord
        for y in np.arange(min_y + 1, max_y):
            new_node = Node()
            new_node.was_added = True
            x = ((max_x - min_x) * (y - min_y)) / (max_y - min_y) + min_x
            new_node.coord = (int(x), int(y))
            new_node.connected.append(prev)
            prev.connected.append(new_node)
            graph.nodes.add(new_node)
            prev = new_node
        end.connected.append(prev)
        prev.connected.append(end)


def get_start_node(graph):
    ends = list(filter(lambda node: len(node.connected) == 1, graph.nodes))
    ends_width = list(map(lambda node: node.width, ends))
    return ends[np.argmax(ends_width)]


def add_tree_attributes(graph):
    start_node = get_start_node(graph)
    graph.start_node = start_node
    start_node.prev = None
    stack = [start_node]
    while stack:
        curr_node = stack.pop()
        curr_node.next = []
        for next_node in curr_node.connected:
            if next_node != curr_node.prev:
                curr_node.next.append(next_node)
                next_node.prev = curr_node
                stack.append(next_node)
                

def fill_width_for_conected(graph):
    start_node = graph.start_node
    dist_before = start_node.width
    stack = [(start_node, [])]
    while stack:
        curr_node, nodes_without_width = stack.pop()
        if len(curr_node.next) > 1 and curr_node.was_added:
            raise Exception("Impossible situation") 
        elif not curr_node.was_added:
            if nodes_without_width:
                start_width = nodes_without_width[0].prev.width
                end_width = curr_node.width
                for i, node in enumerate(nodes_without_width):
                    node.width = end_width + (start_width - end_width) * (i + 1) / (len(nodes_without_width) + 1)
            for next_node in curr_node.next:
                stack.append((next_node, []))
        else:
            for next_node in curr_node.next:
                stack.append((next_node, nodes_without_width + [curr_node]))


def dist(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def connect_end_to_graph(end, graph, new_graph):
    distances, path, _ = Dijkstra(graph, end)
    distances = sorted(list(distances.items()), key=lambda value: value[1])
    shared_node = None
    path_coords = list(map(lambda node: node.coord, new_graph.nodes))
    for node, end_len in distances:
        if node.coord in path_coords:
            shared_node = node
            break

    tmp = shared_node
    prev = new_graph.get_node_by_coord(shared_node.coord) 
    while True:
        tmp = path[tmp]
        new_node = Node()
        new_graph.nodes.add(new_node)
        prev.connected.append(new_node)
        new_node.connected.append(prev)
        prev = new_node
        try:
            new_node.was_added = tmp.was_added
        except:
            new_node.was_added = False
        new_node.coord = tmp.coord
        if path[tmp] == tmp:
            break



def get_max_path_node(distances):
    max_dist, max_node = 0, None
    for node, dist in distances.items():
        if dist > max_dist:
            max_dist = dist
            max_node = node
    return max_node


def get_largest_path_as_graph(graph, remove=True):
    distances, _, _ = Dijkstra(graph, list(graph.nodes)[0])
    max_node1 = get_max_path_node(distances)
    distances, path, ends = Dijkstra(graph, max_node1)
    max_node2 = get_max_path_node(distances)
    new_graph = graph_from_path(graph, path, max_node2)
    for end in ends:
        connect_end_to_graph(end, graph, new_graph)
    if remove:
        remove_small_branches(new_graph, new_graph.get_node_by_coord(max_node1.coord))
    return new_graph
    

class TreeNode():

    def __init__(self, value, parent):
        self.value = value
        self.parent = parent
        self.childrens = dict()

def build_tree(graph, start):
    visited = {node: False for node in graph.nodes}
    visited[start] = True

    root = TreeNode(start, None)
    stack = [root]

    while stack:
        curr = stack.pop()
        for node in curr.value.connected:
            if not visited[node]:
                visited[node] = True
                tree_node = TreeNode(node, curr)
                stack.append(tree_node)
                curr.childrens[tree_node] = node
    return root

def remove_branch(start, next_node, graph):
    start.connected.remove(next_node)
    stack = [(next_node, start)] 
    while stack:
        curr, prev = stack.pop()
        graph.nodes.remove(curr)
        for node in curr.connected:
            if node != prev:
                stack.append((node, curr))

def remove_small_branches(graph, start, min_size=30):
    root = build_tree(graph, start)

    stack = [root]
    node_stack = []
    count_stack = []

    while stack:
        curr = stack.pop()
        node_stack.append((curr, len(curr.childrens)))
        for child in curr.childrens:
            stack.append(child)

    while node_stack:
        curr, size = node_stack.pop()
        if size == 0:
            count_stack.append(1)
        else:    
            tree_lengths = [count_stack.pop() for index in np.arange(size)][::-1]
            sorted_nodes = sorted(zip(map(lambda child: child.value, curr.childrens), tree_lengths), key=lambda value: value[1], reverse=True)
            for node, tree_len in sorted_nodes[1:]:
                if tree_len < min_size:
                    remove_branch(curr.value, node, graph)
            count_stack.append(sorted_nodes[0][1] + 1)


def add_width_to_nodes(graph, dist):
    for node in graph.nodes:
        i,j = node.coord
        node.width = dist[i,j]


class Branch:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.next = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_next_branch(self, next_branch):
        self.next.append(next_branch)


def divide_graph_by_branches(graph):
    start_branch, joints = Branch(), []
    stack = [(graph.start_node, start_branch)]
    while stack:
        curr_node, curr_branch = stack.pop()
        if len(curr_node.next) > 1:
            joints.append(curr_node)
            for node in curr_node.next:
                new_branch = Branch()
                curr_branch.add_next_branch(next_branch)
                stack.append((node, new_branch))
        elif curr_node.next:
            curr_branch.add_node(curr_node)
            stack.append((curr_node.next[0], curr_branch))
        else:
            curr_branch.add_node(curr_node)
    return start_branch, joints






        