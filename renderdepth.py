import os
import Metashape
import numpy as np
from scipy import ndimage


def render_depths(path, xml_path, ply_path):
    """ Uses agisoft to render the absolute depth to the dense point cloud. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    # add xml, cameras, and cloud to chunk
    images = os.listdir(path)
    images = [os.path.join(path, elem) for elem in images]
    chunk.addPhotos(images)  # photos need to be added
    chunk.importCameras(xml_path)
    chunk.importPoints(ply_path)

    for j, camera in enumerate(chunk.cameras):
        print(camera.label)
        if not camera.enabled or camera.transform is None:
            continue

        if os.path.exists(os.path.join(path.replace("0_images", "11_depth"), camera.label) + ".npy"):
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

        np.save(os.path.join(path.replace("0_images", "11_depth"), camera.label), depth_np)
