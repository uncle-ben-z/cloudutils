import open3d as o3d


def create_point_cloud(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])
    return pcd


def set_color(argmax):
    if argmax == 0:
        col = [228 / 255, 26 / 255, 28 / 255]
    elif argmax == 1:
        col = [255 / 255, 127 / 255, 0 / 255]
    elif argmax == 2:
        col = [255 / 255, 255 / 255, 51 / 255]
    elif argmax == 3:
        col = [55 / 255, 126 / 255, 184 / 255]
    elif argmax == 4:
        col = [77 / 255, 175 / 255, 74 / 255]
    elif argmax == 5:
        col = [152 / 255, 78 / 255, 163 / 255]
    else:
        col = [0.7, 0.7, 0.7]

    return col
