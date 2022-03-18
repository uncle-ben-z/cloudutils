import os
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
import xml.etree.ElementTree as ET


def load_agisoft_markers(path):
    """ Loads all markers from agisoft xml. """
    root = ET.parse(path).getroot()

    markers = {}
    for marker in root.iter('marker'):
        if 'marker_id' in marker.attrib.keys():
            continue
        print(marker.attrib['label'])
        idd = np.int32(marker.attrib['id'])
        print(idd)
        label = marker.attrib['label']
        ref = marker.find('reference')
        enabled = True if ref.attrib['enabled'] is None else False
        if True or enabled:
            markers[idd] = np.float32([ref.attrib['x'], ref.attrib['y'], ref.attrib['z']])

    return markers


def export_agisoft_markers(markers, path):
    """ Save marker to agisoft xml file. """
    doc = ET.Element("document")
    doc.tail = "\n"
    doc.text = "\n\t"
    chunk = ET.SubElement(doc, "chunk")
    chunk.tail = "\n"
    chunk.text = "\n\t\t"
    mark = ET.SubElement(chunk, "markers")
    mark.tail = "\n\t"
    mark.text = "\n\t\t\t"

    for key in markers.keys():
        m = ET.SubElement(mark, "marker")
        m.tail = "\n\t\t\t"
        m.text = "\n\t\t\t\t"
        m.attrib['id'] = str(key)
        ref = ET.SubElement(m, "reference")
        ref.tail = "\n\t\t\t"
        ref.attrib['x'] = f"{markers[key][0]:f}"
        ref.attrib['y'] = f"{markers[key][1]:f}"
        ref.attrib['z'] = f"{markers[key][2]:f}"

    xml = ET.tostring(doc, encoding='utf8', method='xml').decode("utf-8")
    print(xml)

    with open(path, 'w') as f:
        f.write(xml)


def extract_markers(ply_path):
    """ Extract markers from point cloud. """
    cloud = PyntCloud.from_file(ply_path)

    defect = np.array(cloud.points["defect"])
    cluster = np.array(cloud.points["cluster"])

    if hasattr(cloud.points, 'normal_x'):
        cloud.points["nx"] = cloud.points["normal_x"]
        cloud.points["ny"] = cloud.points["normal_y"]
        cloud.points["nz"] = cloud.points["normal_z"]

    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([1.0, 0.0, 1.0])

    # only keep control points
    clusters = np.unique(cluster[defect == 1], return_counts=True)

    # loop over control points
    markers = {}
    for c in clusters[0]:
        idxs = np.nonzero(cluster == c)[0]
        subcloud = cloud.select_by_index(idxs)
        markers[c] = subcloud.get_center()

    return markers


def global_registration(source_markers, target_markers):
    """ Finds a transform between source and target point cloud using RANSAC. """
    # source cloud
    source_cloud = o3d.geometry.PointCloud()
    xyz = np.array([list(source_markers[key]) for key in source_markers.keys()])
    source_cloud.points = o3d.utility.Vector3dVector(xyz)

    # target cloud
    target_cloud = o3d.geometry.PointCloud()
    xyz = np.array([list(target_markers[key]) for key in target_markers.keys()])
    target_cloud.points = o3d.utility.Vector3dVector(xyz)

    # Global registration: RANSAC
    # usage of open3d's registration_ransac_based_on... unclear and not returning reasonable results
    num_corres = 3  # three points required for similarity transform
    distance_threshold = 0.1

    # test hypotheses TODO: swap source and target?
    for i in range(1000000):
        # determine correspondence candidates
        corres_cand = np.array([np.random.choice(range(len(source_markers.keys())), num_corres, replace=False),
                                np.random.choice(range(len(target_markers.keys())), num_corres, replace=False)])
        corres_cand = o3d.utility.Vector2iVector(np.int32(corres_cand.T))

        # generate transformation hypothesis
        transform = o3d.pipelines.registration.TransformationEstimationPointToPoint(True).compute_transformation(
            source_cloud, target_cloud, corres_cand)

        # determine inliers
        tmp_cloud = o3d.geometry.PointCloud(source_cloud)
        tmp_cloud = tmp_cloud.transform(transform)
        distances = np.array(target_cloud.compute_point_cloud_distance(tmp_cloud))
        inliers = np.sum(np.where(distances < distance_threshold, 1, 0))

        if inliers > 3:
            break

    # correspondences
    kdtree = o3d.geometry.KDTreeFlann(target_cloud)
    corres = []
    for i in range(len(source_markers)):
        [k, idx, _] = kdtree.search_radius_vector_3d(tmp_cloud.points[i], distance_threshold)
        if k > 0:
            corres.append([i, idx[0]])

    # estimate final transformation
    corres_op3d = o3d.utility.Vector2iVector(np.array(corres))
    transform = o3d.pipelines.registration.TransformationEstimationPointToPoint(True).compute_transformation(
        source_cloud, target_cloud, corres_op3d)

    # remap correspondences
    corres = np.array([[list(source_markers.keys())[elem[0]], list(target_markers.keys())[elem[1]]] for elem in corres])
    print("corres: ", corres)
    print("transform: ")
    print(np.array2string(transform, suppress_small=True))

    return transform, corres


def refine_markers(source_markers):
    # TODO
    pass


def bundle_adjustment():
    # TODO
    pass


def condition_matrix3d(X):
    """ Conditions a homogeneous matrix for numerical stability. """
    t = np.mean(X, axis=1)
    s = np.mean(np.abs(X.T - t), axis=0)
    T = np.array([
        [1 / s[0], 0, 0, -t[0] / s[0]],
        [0, 1 / s[1], 0, -t[1] / s[1]],
        [0, 0, 1 / s[2], -t[2] / s[2]],
        [0, 0, 0, 1],
    ])
    N = T @ X
    return T, N


def homogenize(X):
    """ Converts the columns of a matrix into homogeneous representation. """
    return np.append(X, np.ones((1, X.shape[1])), axis=0)


def helmert_transform(source_markers, target_markers, corres):
    """ Compute the helmert transformation (similarity transformation, 2 1/3 point algorithm). """
    transform = None
    return transform


def estimate_homography3d(X1, X2):
    """ Implementation of the 5-Point algorithm. """
    X1 = homogenize(X1)
    X2 = homogenize(X2)

    # conditioning
    T1, N1 = condition_matrix3d(X1)
    T2, N2 = condition_matrix3d(X2)

    # design matrix
    A = np.empty((0, 16))
    for i in range(N1.shape[1]):
        A = np.append(A, np.array([
            [-N2[3, i] * N1[0, i], -N2[3, i] * N1[1, i], -N2[3, i] * N1[2, i], -N2[3, i] * N1[3, i], 0, 0, 0, 0,
             0, 0, 0, 0, N2[0, i] * N1[0, i], N2[0, i] * N1[1, i], N2[0, i] * N1[2, i], N2[0, i] * N1[3, i]],
            [0, 0, 0, 0, -N2[3, i] * N1[0, i], -N2[3, i] * N1[1, i], -N2[3, i] * N1[2, i], -N2[3, i] * N1[3, i],
             0, 0, 0, 0, N2[1, i] * N1[0, i], N2[1, i] * N1[1, i], N2[1, i] * N1[2, i], N2[1, i] * N1[3, i]],
            [0, 0, 0, 0, 0, 0, 0, 0, -N2[3, i] * N1[0, i], -N2[3, i] * N1[1, i], -N2[3, i] * N1[2, i],
             -N2[3, i] * N1[3, i], N2[2, i] * N1[0, i], N2[2, i] * N1[1, i], N2[2, i] * N1[2, i], N2[2, i] * N1[3, i]],
        ]), axis=0)

    # singular value decomposition
    U, D, V = np.linalg.svd(A)

    # prepare solution, unconditioning
    h_tmp = V[-1, :].reshape(4, 4)
    homography = np.linalg.inv(T2) @ h_tmp @ T1
    homography /= homography[-1, -1]

    return homography
