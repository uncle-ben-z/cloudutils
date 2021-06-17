import cv2
import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
from projection.scene import Scene
from matplotlib import pyplot as plt
from multiprocessing import Pool, Manager
from utils import preload_images, create_point_cloud, set_color


def colorize_point(i, p, normal, idxs, dists, scene, images, colors, number):
    # homogeneous coordinates
    p = np.append(p, np.array([1])).reshape(4, 1)

    # set default color
    colors[i] = [0.7, 0.7, 0.7]

    # empty containers
    angles = []
    distances = []
    probs = {
        "1_crack": [],
        "2_spall": [],
        "3_corr": [],
        "4_effl": [],
        "5_vege": [],
        "6_cp": [],
        "8_background": []
    }

    # loop over neighbors
    for j, idx in enumerate(idxs):
        view = scene.views[idx]

        # viewing angle
        angle = view.world_directive_deviation(normal)

        # only keep acceptable viewing angles
        # if angle < 120 or 240 < angle:
        if angle < 100 or 260 < angle:
            continue

        # project point
        scale = 0.25
        u, v = view.fromWorldToImage(p, scale=scale)

        # continue if outside image
        if u < 0 or scale * view.camera.w <= u or v < 0 or scale * view.camera.h < v:
            continue

        # continue if outside mask
        mask = images[view.name + ".jpg"][-1]
        if mask[int(v[0]), int(u[0])] == 0:
            continue

        # append
        angles.append(angle)
        distances.append(dists[j])
        for k, d in enumerate(probs.keys()):
            img = images[view.name + ".jpg"][k]
            probs[d].append(img[int(v), int(u)])

        # breaking condition
        if len(angles) > 10:
            break

    # compute weights
    # TODO: Method, angles
    distances = np.array(distances)
    angles = np.array(angles)

    if len(distances) == 0:
        colors[i] = [0.7, 0.7, 0.7]
        return

    weight = distances.max() - distances
    weight = weight - weight.min()
    weight /= weight.sum()

    # fuse probabilities
    fus = {}
    for d in probs.keys():
        prob = np.array(probs[d])
        prob = np.sum(prob * weight)
        fus[d] = prob

    colors[i] = set_color(fus)

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


def colorize_point_cloud(pcd_path, img_path, xml_path):
    # load
    pcd = o3d.io.read_point_cloud(pcd_path)
    scene = Scene.from_xml(xml_path)
    images = preload_images(img_path, scale=0.25)

    # create point cloud from views
    view_origins = np.array([view.world_origin for view in scene.views])
    views = create_point_cloud(view_origins)
    views_tree = o3d.geometry.KDTreeFlann(views)

    colors = dict()
    number = len(pcd.points)

    # loop over all points
    for i in tqdm(range(len(pcd.points))):
        # get neigbors
        _, idxs, dists = np.array(views_tree.search_knn_vector_3d(pcd.points[i], 100))

        colorize_point(i, np.array(pcd.points[i]), np.array(pcd.normals[i]),
                       np.int32(idxs),
                       np.float32(dists),
                       scene, images, colors, number)

    colors = dict(colors)

    for k in colors.keys():
        np.asarray(pcd.colors)[[k], :] = colors[k]

    o3d.io.write_point_cloud("./result_cloud.pcd", pcd)

    o3d.visualization.draw_geometries([pcd, views])

    return pcd


if __name__ == "__main__":
    # paths
    pcd_path = "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/points/rebars_smallest.pcd"
    img_path = "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/Christian/VSued_Abplatzung_20210428/0_jpg"
    xml_path = "/media/******/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/maintalbruecke/points/cameras_all.xml"

    pcd = colorize_point_cloud(pcd_path, img_path, xml_path)
