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
