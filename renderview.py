import os
import cv2
import Metashape
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

import xml.etree.ElementTree as ET

from projection import parse_agisoft_xml


def render_views4(model_path, cameras_path, path, is_cloud=False):
    """ Uses agisoft to render views. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    outpath = path.replace("0_images", "13_views")

    xml_root = ET.parse(cameras_path).getroot()
    images = [os.path.join(outpath, view.attrib['label'] + ".png") for view in
              xml_root.iter('camera') if not 'enabled' in view.attrib and not view.find('transform') is None]

    for p in images:
        cv2.imwrite(p, np.zeros((5460, 8192), np.uint8))

    if False and len(images) == len(os.listdir(outpath)):
        print("Views already rendered...")
        return

    chunk.addPhotos(images)  # photos need to be added # TODO: when rendering new images, no images so far known
    chunk.importCameras(cameras_path)

    if is_cloud:
        chunk.importPoints(model_path)
    else:
        chunk.importModel(model_path)

    for j, camera in enumerate(chunk.cameras):
        print(camera.label)
        if not camera.enabled or camera.transform is None:
            continue

        if False and os.path.exists(os.path.join(outpath, camera.label) + ".png"):
            continue

        # render image
        if is_cloud:
            img = chunk.dense_cloud.renderImage(camera.transform, camera.sensor.calibration, add_alpha=False,
                                                point_size=1)
        else:
            img = chunk.model.renderImage(camera.transform, camera.sensor.calibration, add_alpha=False)
        img.save(os.path.join(outpath, camera.label + ".png"))


def render_views3(model_path, cameras_path, out_path, is_cloud=False):
    """ Uses agisoft to render views. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    #    chunk.addPhotos(images)  # photos need to be added # TODO: when rendering new images, no images so far known
    #    chunk.importCameras(cameras_path)

    if is_cloud:
        chunk.importPoints(model_path)
    else:
        chunk.importModel(model_path)

    out_path = out_path.replace("0_images", "13_views")

    images = []
    root = ET.parse(cameras_path).getroot()
    for view in root.iter('camera'):
        # skip in case camera is disabled or no transform available
        if 'enabled' in view.attrib or view.find('transform') is None:
            continue
        path = os.path.join(out_path, view.attrib['label'] + ".png")
        print(path)
        cv2.imwrite(path, np.zeros((5460, 8192), np.uint8))
        images.append(path)

    chunk.addPhotos(images)

    # chunk.importCameras(cameras_path)
    views = parse_agisoft_xml(cameras_path)
    chunk.importCameras(cameras_path)

    for j, camera in enumerate(chunk.cameras):
        print(camera.label)
        img = chunk.model.renderImage(camera.transform, camera.sensor.calibration)
        img.save(os.path.join(out_path, camera.label + ".png"))

    print("Hello")


def render_views(model_path, cameras_path, out_path, is_cloud=False):
    """ Uses agisoft to render views. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    if is_cloud:
        chunk.importPoints(model_path)
    else:
        chunk.importModel(model_path)

    # loop over views
    views = parse_agisoft_xml(cameras_path)
    for label in views.keys():
        # set instrinsics
        calibration = Metashape.Calibration()
        calibration.b1 = views[label].camera.b1
        calibration.b2 = views[label].camera.b2
        calibration.cx = views[label].camera.cx
        calibration.cy = views[label].camera.cy
        calibration.f = views[label].camera.f
        calibration.height = views[label].camera.h
        calibration.k1 = views[label].camera.k1
        calibration.k2 = views[label].camera.k2
        calibration.k3 = views[label].camera.k3
        calibration.k4 = views[label].camera.k4
        calibration.p1 = views[label].camera.p1
        calibration.p2 = views[label].camera.p2
        calibration.p3 = views[label].camera.p3
        calibration.p4 = views[label].camera.p4
        calibration.width = views[label].camera.w

        # undo chunk transform
        camera_transform = chunk.transform.matrix.inv() * Metashape.Matrix(views[label].Rt)

        # render image
        if is_cloud:
            img = chunk.dense_cloud.renderImage(camera_transform.transform, calibration, add_alpha=False, point_size=1)
        else:
            img = chunk.model.renderImage(camera_transform, calibration, add_alpha=False)
        img.save(os.path.join(out_path, label + ".png"))



if __name__ == "__main__":
    print("hello")
    folder_path = "/media/chrisbe/9812080e-2b1a-498a-81e8-99b092601af4/data/referenzobjekte/talbruecke_bruenn/2021_09_08/clean"
    foldername = "pfeiler1_paul"
    cloudname = "Pfeiler_1_07M.ply"  # "Pfeiler_1_591M.ply" #
    camerasname = "cameras_filt.xml"

    render_views(os.path.join(folder_path, foldername, "0_images"), os.path.join(folder_path, foldername, camerasname),
                 os.path.join(folder_path, foldername, cloudname), True)
