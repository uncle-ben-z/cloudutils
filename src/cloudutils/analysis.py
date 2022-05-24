import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud


def cloud_density(cloud_path, mesh_path=None):
    """ Measures the density of a cloud assuming uniform distribution. """
    # load cloud
    cloud = PyntCloud.from_file(cloud_path)
    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([1, 0.706, 0])

    # generate mesh, if necessary
    if mesh_path is None:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8)
        densities = np.asarray(densities)

        # determine threshold for densities
        hist, bin_edges = np.histogram(densities)
        thresh = bin_edges[np.argmax(hist) - 1]

        # remove low densities
        idxs = np.where(densities < thresh)[0]
        mesh.remove_vertices_by_index(idxs)
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

        #o3d.visualization.draw_geometries([mesh])

    else:
        # load available mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)

    # compute resolution
    area = mesh.get_surface_area()
    resolution = len(cloud.points) / (10000 * area) # points per square cm

    return resolution


