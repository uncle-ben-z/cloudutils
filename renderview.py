import os
import cv2
import Metashape
import numpy as np


def render_view(path, xml_path, ply_path):
    """ Uses agisoft to render the absolute depth to the dense point cloud. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    outpath = path.replace("0_images", "13_views")

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

        if os.path.exists(os.path.join(outpath, camera.label) + ".png"):
            continue

        # render image
        img = chunk.dense_cloud.renderImage(camera.transform, camera.sensor.calibration, point_size=1)
        byts = img.tostring()

        # convert to numpy
        img_np = np.copy(np.frombuffer(byts, dtype=np.uint8))
        img_np = img_np.reshape(img.height, img.width, 4)

        cv2.imwrite(os.path.join(outpath, camera.label + ".png"), img_np)
