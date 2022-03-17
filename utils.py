import cv2
import laspy
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.ops import sample_points_from_meshes


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


def mesh2cloud_color_from_vertices(source_path, target_path, count=1000000):
    """ Samples a point cloud from a mesh inferring colors from vertex color (not texture). """
    # Note: sample_points_uniformly unfortunately uses the vertex colors, rather than the colors from the texture.
    mesh = o3d.io.read_triangle_mesh(source_path)
    cloud = mesh.sample_points_uniformly(count, True)
    o3d.io.write_point_cloud(target_path, cloud)


def mesh2cloud(source_path, target_path, count=1000000):
    """ Samples a point cloud from a mesh. """
    meshes = load_objs_as_meshes([source_path], load_textures=True, create_texture_atlas=True)
    points = sample_points_from_meshes(meshes, num_samples=count, return_normals=True, return_textures=True)
    clouds = Pointclouds(points=points[0], normals=points[1], features=points[2])
    IO().save_pointcloud(clouds, target_path)
