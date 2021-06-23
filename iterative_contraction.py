import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from utils import create_graph, remove_duplicates

# load point cloud
pcd = o3d.io.read_point_cloud("crack_1_4M_colorized.pcd")  # "/home/******/Desktop/crack_branched.pcd")

# filter point cloud for cracks
idxs = np.array(np.nonzero(np.array(pcd.colors)[:, 0] > 0.89)[0])
pcd = pcd.select_by_index(idxs)

# find clusters
labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=1, print_progress=True))
uni, count = np.unique(labels, return_counts=True)
uni = uni[count > 3]
count = count[count > 3]

# paint clusters
for lab in uni:
    np.asarray(pcd.colors)[np.nonzero(labels == lab)[0], :] = np.random.random(size=3)

o3d.visualization.draw_geometries([pcd])

lines = []

# loop over components
for k, lab in enumerate(uni):
    pcd_select = pcd.select_by_index(np.nonzero(labels == lab)[0])
    try:
        box_max_extend = np.max(pcd_select.get_oriented_bounding_box().extent)
    except:
        continue
    print("Cluster: ", lab, count[k], box_max_extend)

    if box_max_extend < 0.02:
        continue

    # iteratively contract
    for radius in np.arange(0.001, 0.002, 0.0001):
        kdtree = o3d.geometry.KDTreeFlann(pcd_select)
        tmp = [pcd_select.points[0], pcd_select.points[0]]

        # loop over points and contract
        for i in range(0, len(pcd_select.points)):
            [_, idx, _] = kdtree.search_radius_vector_3d(pcd_select.points[i], radius)
            np.asarray(pcd_select.points)[i, :] = np.mean(np.array(pcd_select.points)[idx, ...], axis=0).reshape(-1, 3)

    # remove duplicate points
    pcd_select = remove_duplicates(pcd_select)

    # create connected graph from point cloud
    G = create_graph(pcd_select)

    # line visualization
    colors = [[1, 0, 0] for i in range(len(G.edges))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pcd_select.points)
    line_set.lines = o3d.utility.Vector2iVector(G.edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    lines.append(line_set)

    # TODO: project point for width measurement


    # simplify graph
    deg = G.degree(G.nodes)
    inter_and_end_nodes = np.array([elem for elem in deg if elem[1] != 2])
    end_nodes = np.array([elem for elem in deg if elem[1] == 1])
    inter_nodes = np.array([elem for elem in deg if elem[1] > 2])

    GG = nx.Graph(G.subgraph(inter_and_end_nodes[:, 0]))

    # no furcations
    if len(inter_nodes) == 0:
        path = nx.shortest_path(G, end_nodes[0, 0], end_nodes[-1, 0])
        pts = [G.nodes[elem]['pos'] for elem in path]
        GG.add_edge(path[0], path[-1], pts=pts)

    # loop over inter nodes
    for source in inter_nodes:
        source = source[0]

        # case: inter node to end node
        for target in end_nodes:
            target = target[0]
            path = nx.shortest_path(G, source, target)

            # if only exactly this inter node in path, add path
            if np.sum(np.isin(inter_nodes[:, 0], path)) == 1:
                pts = [G.nodes[elem]['pos'] for elem in path]
                GG.add_edge(path[0], path[-1], pts=pts)

        # case: inter node to inter node
        for target in inter_nodes:
            target = target[0]
            path = nx.shortest_path(G, source, target)

            # if only exactly these two inter node in path, add path
            if np.sum(np.isin(inter_nodes[:, 0], path)) == 2:
                pts = [G.nodes[elem]['pos'] for elem in path]
                GG.add_edge(path[0], path[-1], pts=pts)

pcd = o3d.io.read_point_cloud(
    "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/points/crack_1_4M.pcd")
lines.append(pcd)
o3d.visualization.draw_geometries(lines)
