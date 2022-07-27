import alphashape
import numpy as np
import open3d as o3d
import networkx as nx
from tqdm import tqdm
from pyntcloud import PyntCloud
from scipy.spatial import distance
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

try:
    from graphutils import remove_duplicates, crack2graph, noncrack2graph, simplify_graph, uniquify_graph_nodes
except:
    from .graphutils import remove_duplicates, crack2graph, noncrack2graph, simplify_graph, uniquify_graph_nodes


def defect2graph(ply_path, graph_path, eps=0.005):
    """ Translate detected defects into networkx graph. """
    cloud = PyntCloud.from_file(ply_path)

    defect = np.array(cloud.points["defect"])
    orig_cluster = np.array(cloud.points["cluster"])
    meta_cluster = np.array(cloud.points["meta_cluster"])

    if hasattr(cloud.points, 'normal_x'):
        cloud.points["nx"] = cloud.points["normal_x"]
        cloud.points["ny"] = cloud.points["normal_y"]
        cloud.points["nz"] = cloud.points["normal_z"]

    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([1.0, 0.0, 1.0])

    G_complete = nx.Graph()

    for c, cluster in enumerate([orig_cluster, meta_cluster]):

        uni, count = np.unique(cluster, return_counts=True)

        # loop over clusters
        for lab in tqdm(uni[1:]):
            idxs = np.nonzero(cluster == lab)[0]

            # select cloud of current cluster
            subcloud = cloud.select_by_index(idxs)

            # get modal (= most plausible) class
            counts = np.bincount(defect[idxs], minlength=7)

            # classes: background, control point, vegetation, efflorescence, corrosion, spalling, crack, exposed rebars

            # determine classes
            if c == 1 and counts[4] > 0 and counts[5] > 0:
                mode_class = 7  # exposed rebar
            elif c == 1 and counts[4] == 0 and counts[5] > 0:
                mode_class = 5
            elif c == 1 and counts[4] > 0 and counts[5] == 0:
                continue
            else:
                mode_class = np.argsort(counts)[-1]

            # determine bounding box
            try:
                box_extend = subcloud.get_oriented_bounding_box().extent
            except:
                continue

            # crack case
            if mode_class == 6 and np.max(box_extend) > 0.02:
                # iteratively contract
                for radius in np.arange(eps / 6, eps / 4, 0.0002):
                    kdtree = o3d.geometry.KDTreeFlann(subcloud)

                    # loop over points and contract
                    for i in range(0, len(subcloud.points)):
                        [_, idx, _] = kdtree.search_radius_vector_3d(subcloud.points[i], radius)

                        # L1 median
                        subcloud_pts = np.array(subcloud.points)[idx, ...]
                        idx_med = np.argmin(np.sum(distance.cdist(subcloud_pts, subcloud_pts), axis=0))

                        try:
                            np.asarray(subcloud.normals)[i, :] = \
                                np.array(subcloud.normals)[idx[idx_med], ...].reshape(-1, 3)

                        except:
                            o3d.visualization.draw_geometries([subcloud])
                        np.asarray(subcloud.points)[i, :] = \
                            np.array(subcloud.points)[idx[idx_med], ...].reshape(-1, 3)

                        # remove duplicate points
                subcloud = remove_duplicates(subcloud)

                # create connected graph from point cloud
                G = crack2graph(subcloud, category=mode_class)

                if len(G.nodes) <= 1:
                    continue

                # simplify graph
                G = simplify_graph(G)

                # unique node ids and compose with complete graph
                GG = uniquify_graph_nodes(G)
                G_complete = nx.compose(G_complete, GG)

            # case: corrosion, spalling, exposed rebar
            elif mode_class != 6 and np.max(box_extend) > 0.01 and mode_class >= 4:
                # reconstruct surface
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(subcloud, depth=9)
                densities = np.asarray(densities)

                # determine threshold for densities
                hist, bin_edges = np.histogram(densities)
                thresh = bin_edges[np.argmax(hist) - 1]

                # skip low density
                if thresh < 3:
                    continue

                # remove low densities
                idxs = np.where(densities < thresh)[0]
                mesh.remove_vertices_by_index(idxs)
                mesh.remove_degenerate_triangles()
                mesh.compute_vertex_normals()

                # o3d.visualization.draw_geometries([mesh, subcloud])

                # get mesh edges
                triangles = np.array(mesh.triangles)
                edges = []
                for i in range(triangles.shape[0]):
                    tmp = np.sort(triangles[i, :])
                    edges.append([tmp[0], tmp[1]])
                    edges.append([tmp[0], tmp[2]])
                    edges.append([tmp[1], tmp[2]])

                # skip for low edge count
                if len(edges) < 6:
                    continue

                edges = np.array(edges)
                vertices = np.array(mesh.vertices)

                # get non-unique edges (since they don't form the border)
                uni, counts = np.unique(edges, return_counts=True, axis=0)
                edges_non_uni = uni[counts != 1, :]

                # create graph
                GG = nx.Graph()
                GG.add_nodes_from(np.arange(len(vertices)))
                GG.add_edges_from(edges)

                # loop over connected components
                S = [GG.subgraph(c).copy() for c in nx.connected_components(GG)]
                for elem in S:
                    # remove non-border edges
                    elem.remove_edges_from(edges_non_uni)

                    # find cycles
                    cycles = nx.cycle_basis(elem)
                    if len(cycles) == 0:
                        continue
                    if len(cycles) > 1:
                        # which one of the cycles is the largest -> keep it
                        # get cycle with most nodes (~ longest cycle)
                        nodes = cycles[np.argmax(np.array([len(c) for c in cycles]))]
                    else:
                        nodes = cycles[0]

                    edges = np.array([nodes, np.roll(nodes, shift=1)]).T

                    # create nodes
                    G = nx.Graph()
                    for node in nodes:
                        G.add_node(node, pos=vertices[node, ...], normal=np.array(mesh.vertex_normals)[node, ...],
                                   category=mode_class)

                    # create edges
                    for edge in edges:
                        src, tar = edge
                        length = np.sqrt(np.sum(np.power(G.nodes[src]["pos"] - G.nodes[tar]["pos"], 2)))
                        points = [G.nodes[src]["pos"], G.nodes[tar]["pos"]]
                        normals = [G.nodes[src]["normal"], G.nodes[tar]["normal"]]
                        G.add_edge(src, tar, weight=length, points=points, normals=normals)

                    if G.size(weight="weight") < 0.06:
                        continue

                    # o3d.visualization.draw_geometries([mesh, subcloud])
                    G = simplify_graph(G)
                    G = uniquify_graph_nodes(G)
                    G_complete = nx.compose(G_complete, G)

            # case: control point
            elif False and mode_class == 1:
                idxs = np.nonzero(cluster == lab)[0]

                # select cloud of current cluster
                subcloud = cloud.select_by_index(idxs)
                center = subcloud.get_center()
                subcloud.points.append(center)
                subcloud.estimate_normals()

                points = np.array(subcloud.points)
                pca = PCA(n_components=2)
                trans = pca.fit_transform(points)

                # Determine the optimized alpha parameter
                # alpha = alphashape.optimizealpha(points)
                alpha_shape = alphashape.alphashape(trans, 2.0)
                bound = np.array(alpha_shape.boundary.coords)
                plt.plot(trans[:, 0], trans[:, 1], '.')
                plt.plot(bound[:, 0], bound[:, 1], 'o')
                plt.show()

                print(center)

                G = nx.Graph()
                G.add_node("_".join(center.astype(str)), pos=subcloud.points[-1], normal=subcloud.normals[-1],
                           category=mode_class)

                # o3d.visualization.draw_geometries([subcloud])
                G_complete = nx.compose(G_complete, G)

    nx.write_gpickle(G_complete, graph_path)
    return


if __name__ == "__main__":
    defect2graph(
        ply_path="/home/******/repos/defect-demonstration/static/uploads/mtb/exprebar_clustered.ply",
        graph_path="/home/******/repos/defect-demonstration/static/uploads/mtb/exprebar_clustered.pickle"
    )
