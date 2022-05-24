import os
from utils import mesh2cloud, apply_transfrom
from render import render_views
from georeferencing import compute_geotransform


if __name__ == "__main__":
    folder_path = "/******"
    foldername = "******"
    modelname = "model/******.obj"
    cloudname = "******.ply"
    model_path = os.path.join(folder_path, foldername, modelname)

    # view point determination
    cameras_path = os.path.join(folder_path, foldername, "cameras.xml")

    # render views (drone flight)
    views_path = os.path.join(folder_path, foldername, "13_views")
    if not os.path.exists(views_path):
        os.mkdir(views_path)
    render_views(model_path=model_path,
                 cameras_path=cameras_path,
                 out_path=views_path)

    # structure-from-motion and dense reconstruction
    target_path = os.path.join(folder_path, foldername, cloudname)
    if False:
        # TODO: test
        # photogrammetric reconstruction
        # https://github.com/agisoft-llc/metashape-scripts
        # https://www.agisoft.com/pdf/metashape_python_api_1_7_4.pdf, page 5
        doc = Metashape.Document()
        chunk = doc.addChunk()

        images = [os.path.join(out_path, elem) for elem in os.listdir(out_path)]
        chunk.addPhotos(images)  # photos need to be added

        chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False)
        chunk.alignCameras()
        # chunk.buildDenseCloud()
        chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation)
        chunk.buildUV(mapping_mode=Metashape.GenericMapping)
        chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)
    else:
        mesh2cloud(model_path, target_path=os.path.join(folder_path, foldername, modelname.replace(".obj", ".ply")))

    # run detection (detection, colorization, clustering)
    pass

    # georeference model
    geotransform = compute_geotransform(folder_path, foldername, cloudname)
    apply_transfrom(cloud_path, geotransform)

    # analyse reco quality
    pass

    # extration of defects
    pass
