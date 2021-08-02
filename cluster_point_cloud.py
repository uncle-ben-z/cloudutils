import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pyntcloud import PyntCloud


def cluster_point_cloud(cloud_path, clusters_path):
    """ Cluster the detected defects and stores them. """
    ply = PyntCloud.from_file(cloud_path)

    # TODO: rethink if defect to color is reasonable
    ply.points.red = 1.0 * ply.points['defect'] / 6.0
    ply.points.green = ply.points['defect'] * 0
    ply.points.blue = ply.points['defect'] * 0

    # load point cloud
    cloud = ply.to_instance("open3d", mesh=False)  # mesh=True by default

    # filter out background
    idxs = np.array(np.nonzero(np.array(cloud.colors)[:, 0] > 0.0)[0])
    cloud = cloud.select_by_index(idxs)

    # find clusters
    labels = np.array(cloud.cluster_dbscan(eps=0.005, min_points=3, print_progress=True))
    uni, count = np.unique(labels, return_counts=True)

    # paint clusters
    for lab in tqdm(uni):
        if lab < 0:
            continue
        # choose modal class for point cloud
        values, counts = np.unique(
            np.uint8(np.asarray(cloud.colors)[np.nonzero(labels == lab)[0], :] * 6), return_counts=True)
        values = values[values != 0]
        counts = values[values != 0]
        col = values[np.argmax(counts)]
        np.asarray(cloud.colors)[np.nonzero(labels == lab)[0], 0] = 1.0 * col / 6

        idxs = np.array(np.nonzero(labels == lab)[0])
        cloud1 = cloud.select_by_index(idxs)
        classes = ["background", "control_point", "vegetation", "efflorescence", "corrosion", "spalling", "crack"]
        o3d.io.write_point_cloud(os.path.join(clusters_path, classes[col] + "_" + str(lab) + ".ply"), cloud1)


if __name__ == "__main__":
    cloud_path = "/home/chrisbe/Desktop/maintal_segment_small1_res.ply"
    clusters_path = "/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/clouds"
    cluster_point_cloud(cloud_path, clusters_path)
