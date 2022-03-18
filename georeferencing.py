import numpy as np
import open3d as o3d


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
    # corres = np.array([[list(source_markers.keys())[elem[0]], list(target_markers.keys())[elem[1]]] for elem in corres])
    print("corres: ", corres)
    print("transform: ")
    print(np.array2string(transform, suppress_small=True))

    return transform, corres


def bundle_adjustment():
    # TODO
    pass


def homogenize(X):
    """ Converts the columns of a matrix into homogeneous representation. """
    return np.append(X, np.ones((1, X.shape[1])), axis=0)


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


def estimate_transformation(source_markers, target_markers, corres):
    """ Compute (similarity) transformation. """
    # source cloud
    source_cloud = o3d.geometry.PointCloud()
    xyz = np.array([list(source_markers[key]) for key in source_markers.keys()])
    source_cloud.points = o3d.utility.Vector3dVector(xyz)

    # target cloud
    target_cloud = o3d.geometry.PointCloud()
    xyz = np.array([list(target_markers[key]) for key in target_markers.keys()])
    target_cloud.points = o3d.utility.Vector3dVector(xyz)

    transform = o3d.pipelines.registration.TransformationEstimationPointToPoint(True).compute_transformation(
        source_cloud, target_cloud, o3d.utility.Vector2iVector(corres))

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
