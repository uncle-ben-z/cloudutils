import os
import Metashape
import numpy as np
from scipy import ndimage


def render_depths(path, xml_path, ply_path):
    """ Uses agisoft to render the absolute depth to the dense point cloud. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    outpath = path.replace("0_images", "10_depth")

    # add xml, cameras, and cloud to chunk
    images = os.listdir(path)
    images = [os.path.join(path, elem) for elem in images]

    if len(images) == len(os.listdir(outpath)):
        print("depth already computed")
        return

    chunk.addPhotos(images)  # photos need to be added
    chunk.importCameras(xml_path)
    chunk.importPoints(ply_path)

    for j, camera in enumerate(chunk.cameras):
        print(camera.label)
        if not camera.enabled or camera.transform is None:
            continue

        if os.path.exists(os.path.join(outpath, camera.label) + ".npy"):
            continue
        # render depth
        depth = chunk.dense_cloud.renderDepth(camera.transform, camera.sensor.calibration, point_size=20)
        depth = depth.convert(" ", "F32")
        bytes = depth.tostring()

        # convert to numpy
        depth_np = np.copy(np.frombuffer(bytes, dtype=np.float32))
        depth_np = depth_np.reshape(depth.height, depth.width)
        depth_np *= chunk.transform.scale

        # filtering
        depth_np = ndimage.median_filter(depth_np, size=4)
        depth_scale = 0.25
        depth_np = depth_np[::int(1 / depth_scale), ::int(1 / depth_scale)]

        np.save(os.path.join(outpath, camera.label), depth_np)


def render_views(model_path, cameras_path, out_path, is_cloud=False):
    """ Uses agisoft to render views. """
    from projection import parse_agisoft_xml
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


def randomize_view(Rt):
    # translation horizontal

    # translation vertical

    # translation distance

    # rotation
    pass
