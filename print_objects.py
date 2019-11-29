import numpy as np
import cv2
from graph import get_start_node


def get_colors(n=3):
    colors = []
    values = np.linspace(0, 255, n)[::-1]
    for r in values:
        for g in values:
            for b in values:
                if r == g and g == b:
                    continue
                colors.append(np.array([r, g, b]))
    return colors


def get_graph_image(graphs, gray=None, width=False):
    if gray is not None:
        img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    else:
        img = np.full((512, 512, 3), 255)
    colors = get_colors()
    for i, (color, graph) in enumerate(zip(colors, graphs)):
        for node in graph.nodes:
            y, x = node.coord
            if width:
                cv2.circle(img, (x, y), int(node.width), color, -1)
            else:
                img[y, x] = color
        start = get_start_node(graph)
        img[start.coord[0], start.coord[1]] = np.array([0, 0, 255])
    return img


def get_ortho_points(point, points, d):
    x, y = point
    if x == 0:
        return [(x, y + d), (x, y - d)]
    if y == 0:
        return [(x - d, y), (x + d, y)]
    xs, ys = points[:, 0], points[:, 1]
    cor = np.corrcoef(xs, ys)[0, 1]
    cor = 0 if np.isnan(cor) else cor
    std_x, std_y = np.std(xs), np.std(ys)
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    if std_x < std_y:
        a = (std_x / std_y) * cor 
        dx = d / np.sqrt(1 + a * a)
        dy = -a * dx
        order = ((start_y > end_y) and dx > 0) or ((start_y < end_y) and dx < 0)
    else:
        a = (std_y / std_x) * cor
        dy = d / np.sqrt(1 + a * a)
        dx = -a * dy
        order = ((start_x > end_x) and dy < 0) or ((start_x < end_x) and dy > 0)
    x1, x2 = x + dx, x - dx
    y1, y2 = y + dy, y - dy
    if order:
        return([(x1, y1), (x2, y2)])
    else:
        return([(x2, y2), (x1, y1)])


def get_poly_of_nodes(nodes, nodes_before=[], nodes_after=[], d=5):
    coords_before = np.array([node.coord for node in nodes_before]).reshape((-1, 2))
    coords_after = np.array([node.coord for node in nodes_after]).reshape((-1, 2))
    coords = np.array([node.coord for node in nodes])
    all_coords = np.vstack([coords_before, coords, coords_after])
    widths = [node.width for node in nodes]
    edges = []
    bias = len(coords_before)
    for i, (coord, width) in enumerate(zip(coords, widths)):
        points = get_ortho_points(coord, all_coords[max(0, i - d + bias):min(len(all_coords)-1, i + d + bias)], width)
        if points:
            edges.append(points)
    edges = np.array(edges)
    edges = np.vstack([edges[:, 0, :], edges[:, 1, :][::-1]])
    edges = np.stack([edges[:, 1], edges[:, 0]], axis=1)
    return edges


def drow_poly(img, poly, color, fill=True):
    pts = np.array(poly, dtype='int32')
    pts = pts.reshape((-1, 1, 2))
    if fill:
        cv2.fillPoly(img, [pts], color, cv2.LINE_8)
    else:
        cv2.polylines(img, [pts], True, color)


tmp_nodes = []
def print_objects_on_graphs(graphs_of_objects, img, fill=False, method='color_per_object'):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        img = np.copy(img)
    colors = get_colors(4)
    colors_dict = {}
    last_color_index = 0
    for graph_of_object in graphs_of_objects:
        stack = [graph_of_object.start_branch]
        while stack:
            branch = stack.pop()
            prev_end_left, prev_end_right = None, None
            for i, found_object in enumerate(branch.found_objects):
                start, end = found_object.description['start_index'], found_object.description['end_index']
                if end - start < 5:
                    tmp_nodes.append(branch.nodes[start:end])
                    continue
                nodes_before = branch.nodes[max(0, start - 5):start]
                nodes_after = branch.nodes[end + 1:min(end + 5, len(branch.nodes) - 1)]
                poly = get_poly_of_nodes(branch.nodes[start:end+1], nodes_before, nodes_after)
                if prev_end_left is not None and prev_end_right is not None:
                    poly[0] = prev_end_left 
                    poly[-1] = prev_end_right
                mid = poly.shape[0] // 2
                if method == 'color_per_object':
                    color = colors[last_color_index]
                    last_color_index = (last_color_index + 1) % len(colors)
                else:
                    if not found_object.description['name'] in colors_dict:
                        colors_dict[found_object.description['name']] = colors[last_color_index]
                        last_color_index = (last_color_index + 1) % len(colors)
                    color = colors_dict[found_object.description['name']]
                coords = np.array([node.coord for node in branch.nodes[start:end+1]])
                img[(coords[:, 0], coords[:, 1])] = np.array(color)
                drow_poly(img, poly, color, fill=fill)
                prev_end_left, prev_end_right = poly[mid - 1], poly[mid]
            for next_branch in branch.next:
                stack.append(next_branch)
    return img