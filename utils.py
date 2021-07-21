import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial.distance import cdist


def create_graph(pcd):
    """ Create connected graph from point cloud. """
    # compute all pairwise distances (upper triangle)
    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    dist = cdist(points, points)
    dist = np.triu(dist, k=0)
    dist[dist == 0] = np.inf

    # create nodes
    G = nx.Graph()
    for i, pt in enumerate(points):
        G.add_node(i, pos=points[i, ...], normal=normals[i, ...])

    # connect until graph is completely connected
    while not nx.is_connected(G):
        src, tar = np.unravel_index(dist.argmin(), dist.shape)
        dist[src, tar] = np.inf

        if not nx.has_path(G, src, tar):
            length = np.sqrt(np.sum(np.power(G.nodes[src]["pos"] - G.nodes[tar]["pos"], 2)))
            G.add_edge(src, tar, weight=length)

    return G


def simplify_graph(G):
    """ Removes intermediate nodes with only two neighbors. """
    deg = G.degree(G.nodes)
    end_nodes = np.array([elem for elem in deg if elem[1] == 1])
    inter_nodes = np.array([elem for elem in deg if elem[1] > 2])
    inter_and_end_nodes = np.array([elem for elem in deg if elem[1] != 2])

    GG = nx.Graph(G.subgraph(inter_and_end_nodes[:, 0]))

    # furcations absent
    if len(inter_nodes) == 0:
        path = nx.shortest_path(G, end_nodes[0, 0], end_nodes[-1, 0])
        points = [G.nodes[elem]['pos'] for elem in path]
        normals = [G.nodes[elem]['normal'] for elem in path]
        GG.add_edge(path[0], path[-1], points=points, normals=normals)

    # furcations present: loop over inter nodes
    for source in inter_nodes:
        source = source[0]

        # case: inter node to end node
        for target in end_nodes:
            target = target[0]
            path = nx.shortest_path(G, source, target)

            # if only exactly this inter node in path, add path
            if np.sum(np.isin(inter_nodes[:, 0], path)) == 1:
                points = [G.nodes[elem]['pos'] for elem in path]
                normals = [G.nodes[elem]['normal'] for elem in path]
                GG.add_edge(path[0], path[-1], points=points, normals=normals)

        # case: inter node to inter node
        for target in inter_nodes:
            target = target[0]
            path = nx.shortest_path(G, source, target)

            # if only exactly these two inter node in path, add path
            if np.sum(np.isin(inter_nodes[:, 0], path)) == 2:
                points = [G.nodes[elem]['pos'] for elem in path]
                normals = [G.nodes[elem]['normal'] for elem in path]
                GG.add_edge(path[0], path[-1], points=points, normals=normals)

    return GG


def uniquify_graph_nodes(G):
    """ Converts node IDs into unique (position-based) IDs. """
    name_mapping = dict()
    for node in G.nodes:
        name_mapping[node] = "_".join(G.nodes[node]["pos"].astype(str))

    G = nx.relabel_nodes(G, name_mapping)
    return G


def remove_duplicates(pcd):
    """ Removes identical points from point cloud. """
    # rounding required
    pcd.points = o3d.utility.Vector3dVector(np.round(np.array(pcd.points), 4))
    uni, idxs = np.unique(np.array(pcd.points), return_index=True, axis=0)
    # create reduced point cloud
    pcd_red = o3d.geometry.PointCloud()
    pcd_red.points = o3d.utility.Vector3dVector(np.array(pcd.points)[idxs, :])
    pcd_red.colors = o3d.utility.Vector3dVector(np.array(pcd.colors)[idxs, :])
    pcd_red.normals = o3d.utility.Vector3dVector(np.array(pcd.normals)[idxs, :])

    return pcd_red


def draw_lines(G):
    """ Draws the nodes and edges of a graph as open3d lines. """
    # map node ids to integers
    mapping = dict(zip(G.nodes, range(len(G.nodes))))
    G = nx.relabel_nodes(G, mapping)

    # determine points and colors
    points = np.array(list(nx.get_node_attributes(G, 'pos').values()))
    colors = [[1, 0, 0] for i in range(len(G.nodes))]

    # create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(G.edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set])
