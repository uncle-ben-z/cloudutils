import os
import cv2
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from projection.scene import Scene
from matplotlib import pyplot as plt
from multiprocessing import Pool, Manager
from utils import create_point_cloud, set_color


def colorize_point(i, p, n, scene, colors):
    # homogeneous coordinates
    p = np.append(p, np.array([1])).reshape(4, 1)

    # determine distances
    distances = np.linalg.norm((scene.origins - p[:3].reshape(3)), axis=1)

    # set default color
    colors[i] = [0.7, 0.7, 0.7]

    # transform from world to camera
    p = scene.transformations @ p

    # 3D to 2D
    p = p.reshape(-1, 4)[..., :-1]
    p[..., 0] = p[..., 0] / p[..., -1]
    p[..., 1] = p[..., 1] / p[..., -1]
    p = p[..., :2]

    # radially distort
    rr = np.linalg.norm(p, axis=1)
    intrins = scene.intrinsics
    px = p[..., 0] * (
            1 + intrins[..., 0] * np.power(rr, 2) +
            intrins[..., 1] * np.power(rr, 4) +
            intrins[..., 2] * np.power(rr, 6) +
            intrins[..., 3] * np.power(rr, 8)) + (
                 intrins[..., 4] * (np.power(rr, 2) + 2 * np.power(p[..., 0], 2)) +
                 2 * intrins[..., 5] * p[..., 0] * p[..., 1])
    py = p[..., 1] * (
            1 + intrins[..., 0] * np.power(rr, 2) +
            intrins[..., 1] * np.power(rr, 4) +
            intrins[..., 2] * np.power(rr, 6) +
            intrins[..., 3] * np.power(rr, 8)) + (
                 intrins[..., 4] * (np.power(rr, 2) + 2 * np.power(p[..., 1], 2)) +
                 2 * intrins[..., 5] * p[..., 0] * p[..., 1])

    # adjust to pixels
    pu = intrins[..., 7] * 0.5 + intrins[..., -2] + px * intrins[..., 6]
    pv = intrins[..., 8] * 0.5 + intrins[..., -1] + py * intrins[..., 6]

    # apply scale
    pu = pu * scene.scale
    pv = pv * scene.scale
    uv = np.int32(np.append(pu.reshape(-1, 1), pv.reshape(-1, 1), axis=1))

    # determine angle between normals (for heuristic visibility check)
    nominator = np.dot(scene.directions, n)
    denominator = np.linalg.norm(n) * np.linalg.norm(scene.directions, axis=1)
    angles = np.degrees(np.arccos(nominator / denominator))

    # TODO: determine angle between viewing direction and point for weighting

    # prepare constraints
    angles = np.where((100 < angles) * (angles < 260), 1, 0)
    uv_mask = np.where((0 <= uv[..., 0]) * (uv[..., 0] < scene.scale * intrins[..., 7]) *
                       (0 <= uv[..., 1]) * (uv[..., 1] < scene.scale * intrins[..., 8]), 1, 0)

    # compute weight
    weight = distances
    weight *= angles
    weight *= uv_mask
    weight = weight.max() - weight
    weight *= angles
    weight *= uv_mask
    weight = np.power(weight, 8)
    weight /= weight.sum()

    # select points
    uv = uv * uv_mask.reshape(-1, 1)

    # get intensities
    intensities = np.array(
        [scene.images[i, uv[i, 1].reshape(-1, 1), uv[i, 0].reshape(-1, 1), :]
         for i in range(uv.shape[0])]).reshape(-1, 8)

    # apply mask
    weight = intensities[..., -1] * weight

    # determine argmax
    probs = intensities[..., :-1] * weight.reshape(-1, 1)
    probs_acc = np.sum(probs, axis=0)
    argmax = np.argmax(probs_acc)

    # get color
    colors[i] = set_color(argmax)




def colorize_point_cloud(pcd_path, xml_path, img_list, result_path="./result_cloud.pcd"):
    """ Paint point cloud. """
    # load and prepare
    pcd = o3d.io.read_point_cloud(pcd_path)
    scene = Scene.from_xml(xml_path)
    scene.prepare_matrices()
    scene.load_images(path_list=None, npy_path="images.npy", scale=0.5)

    colors = dict()

    # loop over all points
    for i in tqdm(range(len(pcd.points))):
        colorize_point(i, np.array(pcd.points[i]), np.array(pcd.normals[i]), scene, colors)

        # store intermediate result
        if i % 10000 == 0:
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
    o3d.io.write_point_cloud(result_path, pcd)

    # visualize point cloud
    view_origins = np.array([view.world_origin for view in scene.views])
    views = create_point_cloud(view_origins)
    o3d.visualization.draw_geometries([pcd, views])

    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Colorize a point cloud based on class probabilities from multiple views.')
    parser.add_argument('pcd_path', type=str, nargs=1,
                        help='point cloud to be colorized')
    parser.add_argument('xml_path', type=str, nargs=1,
                        help='agisoft xml for camera properties')
    parser.add_argument('img_list', type=str, nargs='+',
                        help='list of all folders hosting images for a class')
    parser.add_argument('--result_path', type=str, nargs=1, default="result_cloud.pcd",
                        help='path to resulting point cloud')
    args = parser.parse_args()

    # check for file existence
    if os.path.exists(args.result_path[0]):
        raise RuntimeError(f"File already exists: {args.result_path[0]}")

    # run colorization
    pcd = colorize_point_cloud(args.pcd_path[0], args.xml_path[0], args.img_list, args.result_path[0])
