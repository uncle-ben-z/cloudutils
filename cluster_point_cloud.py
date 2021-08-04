import numpy as np
from tqdm import tqdm
import pandas as pd
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN


def cluster_point_cloud(cloud_path, eps=0.005, min_samples=3):
    """ Cluster the detected defects and store cluster info in cloud properties. """
    # load cloud and get relevant properties
    ply = PyntCloud.from_file(cloud_path)
    xyz = ply.xyz
    defects = np.array(ply.points['defect'])
    clusters = np.int32(np.copy(defects) * 0)

    # loop over all defect classes
    for d in tqdm(np.unique(defects)[1:]):
        # filter for current defect class
        idxs = (defects == d)
        X = xyz[idxs, :]

        # perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # assign unique ids for clusters
        clusters[idxs] = clustering.labels_ + np.max(clusters) + 1

        # assign 0 to invalid clusters
        tmp = clusters[idxs] # tmp variable is needed
        tmp[(clustering.labels_ == -1)] = 0
        clusters[idxs] = tmp

    # set property and save cloud
    ply.points['cluster'] = pd.Series(clusters)
    ply.to_file(cloud_path + ".ply")

    return ply


if __name__ == "__main__":
    cloud_path = "/home/chrisbe/repos/defect-demonstration/static/uploads/2021_07_20__15_19_17/new_result_cloud.ply"
    cluster_point_cloud(cloud_path=cloud_path)
