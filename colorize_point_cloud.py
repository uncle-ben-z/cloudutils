#!/usr/bin/env python
import os
import cv2
import laspy
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from pyntcloud import PyntCloud
from projection.scene import Scene
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from multiprocessing import Pool, Manager
from utils import create_point_cloud, set_color


def compute_sharpness(img, thresh=3):
    """ Computes the local sharpness."""
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # laplacian (w/ diagonal)
    grad = cv2.filter2D(img, cv2.CV_64F, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
    grad = np.uint8((np.abs(grad) / 2040) * 255)

    # compute binary mask
    _, mask = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # compute convex hull
    pts = np.nonzero(mask)
    pts = np.array(pts).T
    hull = ConvexHull(pts)
    pts[:, [0, 1]] = pts[:, [1, 0]]
    mask = cv2.fillPoly(mask, [pts[hull.vertices, :]], color=255)

    # dilate gradient
    grad = cv2.dilate(grad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150)))
    grad = cv2.blur(grad, (100, 100))

    if False:
        ax = plt.subplot(221)
        plt.imshow(img, 'gray')
        plt.subplot(222, sharex=ax, sharey=ax)
        plt.imshow(img)
        plt.plot(pts[hull.vertices, 0], pts[hull.vertices, 1], 'r--', lw=2)
        plt.plot(pts[:, 0], pts[:, 1], '.')
        plt.subplot(223, sharex=ax, sharey=ax)
        plt.imshow(mask)
        plt.subplot(224, sharex=ax, sharey=ax)
        plt.imshow(grad)
        plt.show()

    return mask, grad


def compute_weight(distances, angles, uv_mask):
    """ Computes the weight for each view. """
    # compute weight
    weight = distances
    weight *= angles
    weight *= uv_mask
    weight = weight.max() - weight
    weight *= angles
    weight *= uv_mask
    weight = np.power(weight, 2)
    weight /= weight.sum()
    weight = weight.reshape(-1, 1)
    return weight


def cloud2las(cloud, filepath):
    """ Converts ply point cloud into las (including class information). """
    las = laspy.create(file_version="1.2", point_format=2)
    las.header.scales = np.array([1e-05, 1e-05, 1e-05])
    las.x = np.array(cloud.points.x)
    las.y = np.array(cloud.points.y)
    las.z = np.array(cloud.points.z)
    las.red = np.array(cloud.points.red)
    las.green = np.array(cloud.points.green)
    las.blue = np.array(cloud.points.blue)
    las.classification = np.array(cloud.points['defect'])
    las.write(filepath)

    return


def colorize_point_cloud(ply_path, xml_path, path_list, result_path="./result_cloud.pcd"):
    """ Paint points of point cloud. """
    # load and prepare
    ply = PyntCloud.from_file(ply_path)
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=path_list, npy_path="images.npy", scale=0.5)

    # container for results
    defects = np.zeros((len(ply.points)), np.ubyte)
    confidences = np.zeros((len(ply.points)))

    crack = np.zeros((len(ply.points)))
    spall = np.zeros((len(ply.points)))
    corr = np.zeros((len(ply.points)))
    effl = np.zeros((len(ply.points)))
    vege = np.zeros((len(ply.points)))
    cp = np.zeros((len(ply.points)))
    back = np.zeros((len(ply.points)))

    values = ply.points.values

    # loop over all points
    for i in tqdm(range(len(values))):

        point = values[i]
        uv, uv_mask, distances, angles = scene.point2uvs(point[:3], point[3:6])

        weight = compute_weight(distances, angles, uv_mask)

        # get intensities
        uv = uv * uv_mask.reshape(-1, 1)
        intensities = np.array(
            [scene.images[i, uv[i, 1].reshape(-1, 1), uv[i, 0].reshape(-1, 1), :]
             for i in range(uv.shape[0])]).reshape(-1, 8)

        # determine argmax
        probs = intensities[..., :-1] * weight
        probs_acc = np.sum(probs, axis=0) / 255

        # reverse (for VR)
        probs_acc = probs_acc[::-1]

        if any(np.isnan(probs_acc)):
            argmax = 0
        else:
            argmax = np.argmax(probs_acc)
            confidences[i] = np.diff(np.sort(probs_acc)[-2:])
            crack[i] = probs_acc[6]
            spall[i] = probs_acc[5]
            corr[i] = probs_acc[4]
            effl[i] = probs_acc[3]
            vege[i] = probs_acc[2]
            cp[i] = probs_acc[1]
            back[i] = probs_acc[0]

        defects[i] = argmax

    # set properties
    ply.points["defect"] = pd.Series(defects, dtype=np.ubyte)
    ply.points["confidence"] = pd.Series(confidences)
    ply.points["background"] = pd.Series(back)
    ply.points["control_point"] = pd.Series(cp)
    ply.points["vegetation"] = pd.Series(vege)
    ply.points["efflorescence"] = pd.Series(effl)
    ply.points["corrosion"] = pd.Series(corr)
    ply.points["spalling"] = pd.Series(spall)
    ply.points["crack"] = pd.Series(crack)

    # store point cloud
    print(result_path)
    ply.to_file(result_path)

    # visualize point cloud
    if False:
        o3d_ply = PyntCloud.from_instance("open3d", ply)
        view_origins = np.array([view.world_origin for view in scene.views])
        views = create_point_cloud(view_origins)
        o3d.visualization.draw_geometries([o3d_ply, views])

    return ply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Colorize a point cloud based on class probabilities from multiple views.')
    parser.add_argument('--pcd_path', type=str, nargs=1,
                        help='point cloud to be colorized', default=[
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/points/crack_1_4M.pcd"])
    parser.add_argument('--xml_path', type=str, nargs=1,
                        help='agisoft xml for camera properties', default=[
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/points/cameras_all.xml"])
    parser.add_argument('--img_list', type=str, nargs='+',
                        help='list of all folders hosting images for a class', default=[
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/1_crack",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/2_spall",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/3_corr",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/4_effl",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/5_vege",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/6_cp",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/8_background",
            "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/9_mask"])
    parser.add_argument('--result_path', type=str, nargs=1, help='path to resulting point cloud',
                        default=["result_cloud.pcd"])
    args = parser.parse_args()

    # check for file existence
    #    if os.path.exists(args.result_path[0]):
    #        raise RuntimeError(f"File already exists: {args.result_path[0]}")

    # run colorization
    ply = colorize_point_cloud(args.pcd_path[0], args.xml_path[0], args.img_list, args.result_path[0])
