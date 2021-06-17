import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing
from matplotlib import pyplot as plt


def preload_images(img_path, scale=1):
    images = {}

    for f in tqdm(os.listdir(img_path)):
        load_image(f, images, img_path, scale)

        if False:
            plt.subplot(121)
            plt.imshow(images[f]["2_spall"])
            plt.subplot(122)
            plt.imshow(images[f]["9_mask"])
            plt.show()

    return images


def load_image(f, images, img_path, scale):
    images[f] = []
    for d in ["1_crack", "2_spall", "3_corr", "4_effl", "5_vege", "6_cp", "8_background"]:
        img = cv2.imread(os.path.join(img_path.replace("0_jpg", d), f), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        img = cv2.resize(img, (int(scale*w), int(scale*h)))
        images[f].append(img)
    try:
        img = cv2.imread(os.path.join(img_path.replace("0_jpg", "9_mask"), f.replace(".jpg", ".png")),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (int(scale*w), int(scale*h)))
    except:
        img = (img * 0) + 255
    images[f].append(np.uint8(img))


def create_point_cloud(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    return pcd


def set_color(fus):
    key = max(fus, key=fus.get)

    if key == "1_crack":
        col = [0, 0, 0]
    elif key == "2_spall":
        col = [228 / 255, 26 / 255, 28 / 255]
    elif key == "3_corr":
        col = [255 / 255, 127 / 255, 0 / 255]
    elif key == "4_effl":
        col = [55 / 255, 126 / 255, 184 / 255]
    elif key == "5_vege":
        col = [55 / 255, 126 / 255, 184 / 255]
    elif key == "6_gc":
        col = [152 / 255, 78 / 255, 163 / 255]
    else:
        col = [0.7, 0.7, 0.7]

    return col
