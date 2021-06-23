import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial.distance import cdist


def create_graph(pcd):
    """ Create connected graph from point cloud. """
    # compute all pairwise distances (upper triangle)
    points = np.array(pcd.points)
    dist = cdist(points, points)
    dist = np.triu(dist, k=0)
    dist[dist == 0] = np.inf

    # create nodes
    G = nx.Graph()
    for i, pt in enumerate(points):
        G.add_node(i, pos=points[i, ...])

    # connect until graph is completely connected
    while not nx.is_connected(G):
        src, tar = np.unravel_index(dist.argmin(), dist.shape)
        dist[src, tar] = np.inf

        if not nx.has_path(G, src, tar):
            length = np.sqrt(np.sum(np.power(G.nodes[src]["pos"] - G.nodes[tar]["pos"], 2)))
            G.add_edge(src, tar, weight=length)

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
