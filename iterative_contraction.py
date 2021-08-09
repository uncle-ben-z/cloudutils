import numpy as np
import open3d as o3d
import networkx as nx
from tqdm import tqdm
from pyntcloud import PyntCloud
from matplotlib import pyplot as plt
from utils import remove_duplicates, crack2graph, noncrack2graph, simplify_graph, uniquify_graph_nodes, draw_lines


def defect2graph(ply_path, graph_path, eps=0.005):
    cloud = PyntCloud.from_file(ply_path)

    cluster = np.array(cloud.points["cluster"])
    defect = np.array(cloud.points["defect"])

    # convert to open3d cloud;
    # TODO: check normals
    try:
        cloud.points["nx"] = cloud.points["normal_x"]
        cloud.points["ny"] = cloud.points["normal_y"]
        cloud.points["nz"] = cloud.points["normal_z"]
    except:
        pass
    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([1.0, 0.0, 0.0])

    uni, count = np.unique(cluster, return_counts=True)

    G_complete = nx.Graph()

    # loop over clusters
    for lab in tqdm(uni[1:]):
        idxs = np.nonzero(cluster == lab)[0]
        other_idxs = np.nonzero(cluster != lab)[0]

        # select cloud of current cluster
        subcloud = cloud.select_by_index(idxs)

        # get modal (= most plausible) class
        mode_class = np.argmax(np.bincount(defect[idxs]))

        # determine bounding box
        try:
            box_extend = subcloud.get_oriented_bounding_box().extent
        except:
            continue

        # case: crack
        if mode_class == 6 and np.max(box_extend) > 0.02:
            # iteratively contract
            for radius in np.arange(0.001, eps, 0.0002):
                kdtree = o3d.geometry.KDTreeFlann(subcloud)

                # loop over points and contract
                for i in range(0, len(subcloud.points)):
                    [_, idx, _] = kdtree.search_radius_vector_3d(subcloud.points[i], radius)
                    try:
                        np.asarray(subcloud.normals)[i, :] = \
                            np.mean(np.array(subcloud.normals)[idx, ...], axis=0).reshape(-1, 3)
                    except:
                        o3d.visualization.draw_geometries([subcloud])
                    np.asarray(subcloud.points)[i, :] = \
                        np.mean(np.array(subcloud.points)[idx, ...], axis=0).reshape(-1, 3)

            # remove duplicate points
            subcloud = remove_duplicates(subcloud)

            # create connected graph from point cloud
            G = crack2graph(subcloud, category=mode_class)

            # simplify graph
            G = simplify_graph(G)

            # unique node ids and compose with complete graph
            GG = uniquify_graph_nodes(G)
            G_complete = nx.compose(G_complete, GG)

        # case: non-crack
        elif mode_class != 6 and np.max(box_extend) > 0.01:
            # TODO: clustering?
            remaining_cloud = cloud.select_by_index(other_idxs)
            kdtree = o3d.geometry.KDTreeFlann(remaining_cloud)

            border_idxs = []

            # determine nearest neighbors in surrounding cloud
            for i in range(0, len(subcloud.points)):
                [_, idx, _] = kdtree.search_knn_vector_3d(subcloud.points[i], knn=1)
                border_idxs.append(idx[0])

            # select points of border cloud
            border_cloud = remaining_cloud.select_by_index(border_idxs)

            # remove duplicates
            border_cloud = remove_duplicates(border_cloud)

            # create graph
            G = noncrack2graph(border_cloud, category=mode_class)
            G = uniquify_graph_nodes(G)
            G_complete = nx.compose(G_complete, G)

            # draw_lines(G, border_cloud)

    nx.write_gpickle(G_complete, graph_path)
    return


if __name__ == "__main__":
    defect2graph(
        ply_path="../../static/uploads/2021_07_20__15_19_17/old_result_cloud.ply__.ply",
        graph_path="../../static/uploads/2021_07_20__15_19_17/graph_complete.pickle")
