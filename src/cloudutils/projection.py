import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# cf. nas/repos/cloud-colorizer/projection/... and repos/projection_utils/...

class View:
    """ View class representing the extrinsics and intrinsics of an image. """

    def __init__(self, label, Rt, camera):
        self.label = label
        self.Rt = Rt
        self.camera = camera

    def __str__(self):
        txt = "Label: \t" + self.label
        txt += "\nRt: \n"
        txt += str(self.Rt)
        txt += "\nCamera: \t" + str(self.camera)
        return txt

    @property
    def world_origin(self):
        """ Returns the origin in world coordinates. """
        return (self.Rt @ np.array([0, 0, 0, 1]).T)[:3]

    @property
    def viewing_direction(self):
        # TODO: test
        """ Returns the viewing direction in the world coordinate system. """
        return self.world_origin - (self.Rt @ np.array([0, 0, -1, 1]).T)[:3]

    def viewing_deviation(self, normal):
        # TODO: test
        """ Computes the angular deviation between the world viewing direction and a normal. """
        nominator = np.dot(normal, self.world_viewing_direction)
        denominator = np.linalg.norm(normal) * np.linalg.norm(self.world_viewing_direction)
        return np.degrees(np.arccos(nominator / denominator))

    def _world2camera(self, p):
        p = np.append(p, 1)
        """ Transform from world to camera coordinate system. """
        return np.linalg.inv(self.Rt) @ p

    def _camera2image(self, p, scale=1):
        """ Transform from camera to image coordinates. """
        u, v = self.camera._camera2image(p, scale)
        return u, v

    def _world2image(self, p, scale=1):
        p = self._world2camera(p)
        return self._camera2image(p, scale)

    def project(self, p, scale=1):
        return np.int32(self._world2image(p, scale))


class Camera:
    """ Camera class representing the intrinsic properties of a camera. """

    def __init__(self, f, k1, k2, k3=0, k4=0, p1=0, p2=0, p3=0, p4=0, b1=0, b2=0, cx=0, cy=0, w=0, h=0,
                 pixel_width=1., pixel_height=1., focal_length=1.):
        self.f = f
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.b1 = b1
        self.b2 = b2
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.focal_length = focal_length

    def __str__(self):
        out = "\nf: \t" + str(self.f)
        out += "\ncx: \t" + str(self.cx)
        out += "\ncy: \t" + str(self.cy)
        out += "\nk1: \t" + str(self.k1)
        out += "\nk2: \t" + str(self.k2)
        out += "\nk3: \t" + str(self.k3)
        out += "\nk4: \t" + str(self.k4)
        out += "\np1: \t" + str(self.p1)
        out += "\np2: \t" + str(self.p2)
        out += "\np3: \t" + str(self.p3)
        out += "\np4: \t" + str(self.p4)
        out += "\nb1: \t" + str(self.b1)
        out += "\nb2: \t" + str(self.b2)
        out += "\nw: \t" + str(self.w)
        out += "\nh: \t" + str(self.h)
        return out

    def _radial_distortion(self, x, y):
        """ Corrects the radial distortion. """
        r = np.sqrt(x ** 2 + y ** 2)
        x = x * (1 + self.k1 * r ** 2 + self.k2 * r ** 4 + self.k3 * r ** 6 + self.k4 * r ** 8) + (
                self.p1 * (r ** 2 + 2 * x ** 2) + 2 * self.p2 * x * y)
        y = y * (1 + self.k1 * r ** 2 + self.k2 * r ** 4 + self.k3 * r ** 6 + self.k4 * r ** 8) + (
                self.p2 * (r ** 2 + 2 * y ** 2) + 2 * self.p1 * x * y)

        # 1 0.997122 1 0.997102 1 0.997830 1 0.995084 1 944.612335
        return x, y

    def _camera2image(self, p, scale=1):
        """ Transforms point from camera to image coordinates. """
        # project
        x, y, z = p[:3]
        x /= z
        y /= z

        # correct radial distortion
        x, y = self._radial_distortion(x, y)

        # transform to image coordinates
        u = self.w * 0.5 + self.cx + x * self.f + x * self.b1 + y * self.b2
        v = self.h * 0.5 + self.cy + y * self.f

        # apply scale
        u, v = scale * u, scale * v
        return u, v


def parse_bundler(path, w=0, h=0):
    """ Parse bundler cameras file. """
    # parse image labels
    labels = [label.split('.')[0] for label in open(os.path.join(os.path.dirname(path), "list.txt"), 'r').readlines()]

    with open(path, 'r') as f:
        f.readline()
        num_cameras, num_points = np.int32(f.readline().split())

        # parse views
        views = {}
        for i in range(num_cameras):
            focal, k1, k2 = np.float32(f.readline().split())
            Rt = np.eye(4)
            Rt[0, :3] = np.float32(f.readline().split())
            Rt[1, :3] = np.float32(f.readline().split())
            Rt[2, :3] = np.float32(f.readline().split())
            Rt[3, :3] = -Rt[:3, :3].T @ np.float32(f.readline().split())

            Rt = Rt.T
            Rt[:, 1:3] *= -1

            views[labels[i]] = View(labels[i], Rt, Camera(focal, k1, k2, w=w, h=h))

    return views


def parse_agisoft_xml(path):
    """ Parse agisoft cameras xml. """
    root = ET.parse(path).getroot()

    # parse intrinsics
    intrinsics = {}
    # loop over intrinsics
    for sensor in root.iter('sensor'):
        id = np.int32(sensor.attrib['id'])
        calib = sensor.find('calibration')
        f = float(calib.find('f').text)
        cx = float(calib.find('cx').text) if calib.find('cx') is not None else 0
        cy = float(calib.find('cy').text) if calib.find('cy') is not None else 0
        k1 = float(calib.find('k1').text) if calib.find('k1') is not None else 0
        k2 = float(calib.find('k2').text) if calib.find('k2') is not None else 0
        k3 = float(calib.find('k3').text) if calib.find('k3') is not None else 0
        k4 = float(calib.find('k4').text) if calib.find('k4') is not None else 0
        p1 = float(calib.find('p1').text) if calib.find('p1') is not None else 0
        p2 = float(calib.find('p2').text) if calib.find('p2') is not None else 0
        p3 = float(calib.find('p1').text) if calib.find('p1') is not None else 0
        p4 = float(calib.find('p2').text) if calib.find('p2') is not None else 0
        b1 = float(calib.find('b1').text) if calib.find('p1') is not None else 0
        b2 = float(calib.find('b2').text) if calib.find('p2') is not None else 0
        w = int(calib.find('resolution').attrib['width'])
        h = int(calib.find('resolution').attrib['height'])

        for prop in sensor.iter('property'):
            if prop.attrib['name'] == "pixel_width":
                pixel_width = float(prop.attrib['value'])
            elif prop.attrib['name'] == "pixel_height":
                pixel_height = float(prop.attrib['value'])
            elif prop.attrib['name'] == "focal_length":
                focal_length = float(prop.attrib['value'])

        # store intrinsics
        intrinsics[id] = Camera(f, k1, k2, k3, k4, p1, p2, p3, p4, b1, b2, cx, cy, w, h,
                                pixel_width, pixel_height, focal_length)

    # parse chunk transform
    """ Get the transform for the agisoft chunk """
    chunk_transform = np.eye(4)
    # rotation
    rotation = root.find('./chunk/transform/rotation').text
    rotation = np.array(rotation.split())
    rotation = rotation.astype(np.float).reshape((3, 3))
    rotation *= float(root.find('./chunk/transform/scale').text)  # TODO: right?
    chunk_transform[:3, :3] = rotation
    # translation
    translation = root.find('./chunk/transform/translation').text
    translation = np.array(translation.split())
    translation = translation.astype(np.float)
    chunk_transform[:3, 3] = translation

    # parse views
    views = {}
    # find images and corresponding views
    for view in root.iter('camera'):
        # skip in case camera is disabled or no transform available
        if 'enabled' in view.attrib or view.find('transform') is None:
            continue

        # get the inverted camera matrix
        Rt = view.find('transform').text
        Rt = np.float32(Rt.split()).reshape((4, 4))
        # undo chunk transform
        Rt = chunk_transform @ Rt

        label = view.attrib['label']
        views[label] = View(label, Rt, intrinsics[np.int32(view.attrib['sensor_id'])])

    return views

