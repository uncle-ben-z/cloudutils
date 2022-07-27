import os
import cv2
import laspy
import Metashape
import numpy as np
import pandas as pd
import open3d as o3d
from pyntcloud import PyntCloud
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

def image2cloud(img, lab=None):
    h, w, _ = img.shape

    # x and y coordinates
    x, y = np.arange(w) - w / 2, np.arange(h) - h / 2
    xx, yy = np.meshgrid(x, y)

    # map image to cloud
    data = {}
    data['x'] = xx.flatten()
    data['y'] = yy.flatten()
    data['z'] = np.zeros((h * w,))
    data['nx'] = np.zeros((h * w,))
    data['nx'] = np.zeros((h * w,))
    data['nz'] = np.ones((h * w,))
    data['red'] = img[..., 0].flatten()
    data['green'] = img[..., 1].flatten()
    data['blue'] = img[..., 2].flatten()
    if lab is not None:
        data['defect'] = lab.flatten()

    points = pd.DataFrame(data)
    ply = PyntCloud(points)
    return ply

def apply_transfrom(cloud_path):
    # TODO: likely with pyncloud
    pass


def export_agisoft_model(chunk, path, name):
    """ Export mesh from agisoft chunk. """
    obj = "mtllib " + name + ".mtl\n"
    obj += "usemtl " + name + "\n"

    model = chunk.model
    transformation = np.array(chunk.transform.matrix).reshape(4, 4)

    for vertex in model.vertices:
        # print(vertex.coord)
        # TODO: apply chunk transform!
        coord = np.array([vertex.coord[0], vertex.coord[1], vertex.coord[2], 1])
        coord = transformation @ coord

        obj += f"v {coord[0]} {coord[1]} {coord[2]} {vertex.color[0] / 255} {vertex.color[1] / 255} {vertex.color[2] / 255}\n"

    for tex_vertex in model.tex_vertices:
        obj += f"vt {tex_vertex.coord[0]} {tex_vertex.coord[1]}\n"

    for face in model.faces:
        obj += f"f {face.vertices[0] + 1}/{face.tex_vertices[0] + 1} {face.vertices[1] + 1}/{face.tex_vertices[1] + 1} {face.vertices[2] + 1}/{face.tex_vertices[2] + 1}\n"

    with open(os.path.join(path, name + ".obj"), 'w') as f:
        f.write(obj)

    mtl = "newmtl Solid\n"
    mtl += "Ka 1.0 1.0 1.0\n"
    mtl += "Kd 1.0 1.0 1.0\n"
    mtl += "Ks 0.0 0.0 0.0\n"
    mtl += "d 1.0\n"
    mtl += "Ns 0.0\n"
    mtl += "illum 0\n"
    mtl += "\n"
    mtl += "newmtl " + name + "\n"
    mtl += "Ka 1.0 1.0 1.0\n"
    mtl += "Kd 1.0 1.0 1.0\n"
    mtl += "Ks 0.0 0.0 0.0\n"
    mtl += "d 1.0\n"
    mtl += "Ns 0.0\n"
    mtl += "illum 0\n"
    mtl += "map_Kd " + name + ".jpg\n"

    with open(os.path.join(path, name + ".mtl"), 'w') as f:
        f.write(mtl)

    for texture in model.textures:
        h, w, c = texture.image().height, texture.image().width, texture.image().cn
        img = np.fromstring(texture.image().tostring(), dtype=np.uint8)
        img = img.reshape(h, w, c)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(path, name + ".jpg"), img)


def cast_ray(coord, camera, chunk):
    """ Use agisoft function for ray casting. """
    pt2D = Metashape.Vector((int(coord[0]), int(coord[1])))
    # sdiff 2D -> 3D
    # camera origin in chunk world
    center = camera.center
    # pixel in 3D world coordinates
    dot = camera.calibration.unproject(pt2D)
    # transform point into chunk world
    dot = camera.transform * Metashape.Vector((dot[0], dot[1], dot[2], 1.0))

    # ray casting
    intersect = chunk.model.pickPoint(center[:3], dot[:3])

    if intersect is None:
        return None

    # transform intersect from chunk to world coordinates
    coords = chunk.transform.matrix * Metashape.Vector((intersect[0], intersect[1], intersect[2], 1.0))
    coords = [coords[0], coords[1], coords[2]]
    return coords
