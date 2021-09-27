import cv2
import laspy
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt


def compute_sharpness(img, thresh=3):
    """ Computes the local sharpness."""
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # laplacian (w/ diagonal)
    grad = cv2.filter2D(img, cv2.CV_64F, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
    grad = np.uint8((np.abs(grad) / 2040) * 255)

    # dilate gradient
    grad = cv2.dilate(grad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150)))
    grad = cv2.blur(grad, (100, 100))

    return grad


def ply2las(cloud, filepath):
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
    las.pt_src_id = np.array(cloud.points['cluster'], np.uint16)
    las.intensity = np.array(cloud.points['confidence'] * 255, np.uint8)
    las.write(filepath)
    return
