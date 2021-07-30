#!/usr/bin/env python
import os
import cv2
import laspy
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
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
    las = laspy.create(file_version="1.2", point_format=3) # TODO what is version and point format?
    las.x = np.array(cloud.points.x)
    las.y = np.array(cloud.points.y)
    las.z = np.array(cloud.points.z)
    las.classification = np.array(cloud.points['class'])
    las.write(filepath)
    return


def colorize_point_cloud(pcd_path, xml_path, path_list, result_path="./result_cloud.pcd"):
    """ Paint points of point cloud. """
    # load and prepare
    pcd = o3d.io.read_point_cloud(pcd_path)
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=path_list, npy_path="images.npy", scale=0.5)

    colors = dict()

    # loop over all points
    for i in tqdm(range(len(pcd.points))):

        uv, uv_mask, distances, angles = scene.point2uvs(np.array(pcd.points[i]), np.array(pcd.normals[i]))

        weight = compute_weight(distances, angles, uv_mask)

        # get intensities
        uv = uv * uv_mask.reshape(-1, 1)
        intensities = np.array(
            [scene.images[i, uv[i, 1].reshape(-1, 1), uv[i, 0].reshape(-1, 1), :]
             for i in range(uv.shape[0])]).reshape(-1, 8)

        # determine argmax
        probs = intensities[..., :-1] * weight
        probs_acc = np.sum(probs, axis=0)
        if any(np.isnan(probs_acc)):
            argmax = 6
        else:
            argmax = np.argmax(probs_acc)

        # get color
        colors[i] = set_color(argmax)

        # store intermediate result
        if False and i % 10000 == 0:
            pcd = o3d.io.read_point_cloud(pcd_path)

            tmp = dict(colors)

            for k in tmp.keys():
                np.asarray(pcd.colors)[[k], :] = tmp[k]

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            ctr.rotate(0, -490.0)
            ctr.rotate(-200, 0)
            ctr.set_zoom(0.45)
            img = np.array(vis.capture_screen_float_buffer(True))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"./images/an_image_{i}.png", np.uint8(img * 255))

    # paint point cloud
    colors = dict(colors)
    for k in colors.keys():
        np.asarray(pcd.colors)[[k], :] = colors[k]

    # store point cloud
    print(result_path)
    o3d.io.write_point_cloud(result_path, pcd)

    # visualize point cloud
    view_origins = np.array([view.world_origin for view in scene.views])
    views = create_point_cloud(view_origins)
    o3d.visualization.draw_geometries([pcd, views])

    return pcd


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
    pcd = colorize_point_cloud(args.pcd_path[0], args.xml_path[0], args.img_list, args.result_path[0])
