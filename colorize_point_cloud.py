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
from matplotlib import pyplot as plt
from multiprocessing import Pool, Manager
from utils import create_point_cloud, set_color


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
