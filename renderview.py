import os
import cv2
import Metashape
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def render_views(model_path, cameras_path, path, is_cloud=False):
    """ Uses agisoft to render views. """
    doc = Metashape.Document()
    chunk = doc.addChunk()

    outpath = path.replace("0_images", "13_views")

    # add xml, cameras, and cloud to chunk
    images = os.listdir(path)
    images = [os.path.join(path, elem) for elem in images]

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
            img = chunk.dense_cloud.renderImage(camera.transform, camera.sensor.calibration, point_size=1)
        else:
            img = chunk.model.renderImage(camera.transform, camera.sensor.calibration)
        img.save(os.path.join(outpath, camera.label + ".png"))
        byts = img.tostring()

        # convert to numpy
        img_np = np.copy(np.frombuffer(byts, dtype=np.uint8))
        img_np = img_np.reshape(img.height, img.width, 4)

        #plt.imshow(img_np[..., :3])
        #plt.show()

        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(outpath, camera.label + ".png"), img_np)

