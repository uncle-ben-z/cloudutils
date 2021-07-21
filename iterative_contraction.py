import numpy as np
import open3d as o3d
import networkx as nx
from tqdm import tqdm
from .utils import remove_duplicates, create_graph, simplify_graph, uniquify_graph_nodes, draw_lines


def contract_point_cloud(pcd_path, graph_path, eps=0.005):
    # load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)

    # filter point cloud for cracks
    idxs = np.array(np.nonzero(np.array(pcd.colors)[:, 1] < 0.2)[0])
    pcd = pcd.select_by_index(idxs)

    # find clusters
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=1, print_progress=True))
    uni, count = np.unique(labels, return_counts=True)
    uni = uni[count > 3]
    count = count[count > 3]

    # paint clusters
    for lab in uni:
        np.asarray(pcd.colors)[np.nonzero(labels == lab)[0], :] = np.random.random(size=3)

    o3d.visualization.draw_geometries([pcd])

    G_complete = nx.Graph()

    # loop over components
    for k, lab in enumerate(tqdm(uni)):
        pcd_select = pcd.select_by_index(np.nonzero(labels == lab)[0])
        try:
            box_max_extend = np.max(pcd_select.get_oriented_bounding_box().extent)
        except:
            continue

        if box_max_extend < 0.02 or box_max_extend > 0.3:
            continue

        # iteratively contract
        for radius in np.arange(0.001, eps, 0.0002):
            kdtree = o3d.geometry.KDTreeFlann(pcd_select)
            tmp = [pcd_select.points[0], pcd_select.points[0]]

            # loop over points and contract
            for i in range(0, len(pcd_select.points)):
                [_, idx, _] = kdtree.search_radius_vector_3d(pcd_select.points[i], radius)
                np.asarray(pcd_select.points)[i, :] = \
                    np.mean(np.array(pcd_select.points)[idx, ...], axis=0).reshape(-1, 3)

        # remove duplicate points
        pcd_select = remove_duplicates(pcd_select)

        # create connected graph from point cloud
        G = create_graph(pcd_select)

        # simplify graph
        GG = simplify_graph(G)

        # unique node ids and compose with complete graph
        GG = uniquify_graph_nodes(GG)
        G_complete = nx.compose(G_complete, GG)

    # save graph
    nx.write_gpickle(G_complete, graph_path)

    draw_lines(G_complete)

    return G_complete


if __name__ == "__main__":
    contract_point_cloud(
        pcd_path="../../static/uploads/2021_07_20__15_19_17/result_cloud.pcd",
        graph_path="graphs/graph_complete.pickle")
