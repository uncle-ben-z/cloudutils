import os
import math
import open3d as o3d
import pandas as pd

from pyntcloud import PyntCloud

subclouds = []

def store_subcloud(node, node_info, cloud, path, count):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode) and isinstance(node,
                                                                        o3d.geometry.OctreeInternalPointNode) and len(
        node.indices) < count \
            or isinstance(node, o3d.geometry.OctreeLeafNode) and isinstance(node,
                                                                            o3d.geometry.OctreePointColorLeafNode):
        subcloud = cloud.select_by_index(node.indices)
        subcloudname = os.path.join(f"subcloud_{node_info.depth}_{node_info.child_index}_{len(node.indices)}.ply")

        global subclouds
        subclouds.append(subcloudname)

        o3d.io.write_point_cloud(
            os.path.join(path, subcloudname), subcloud)
        early_stop = True

    return early_stop


def create_octree(folder_path, foldername, cloudname, count=100000):
    cloud = o3d.io.read_point_cloud(os.path.join(folder_path, foldername, cloudname))

    # estimate required depth
    depth = int(math.log(len(cloud.points) / count) / math.log(8)) + 2

    global subclouds
    subclouds = []

    # create octree
    octree = o3d.geometry.Octree(max_depth=depth)
    octree.convert_from_point_cloud(cloud, size_expand=0.00)
    octree.traverse(
        lambda node, node_info: store_subcloud(node, node_info, cloud,
                                               os.path.join(folder_path, foldername, "12_octree"), count))
    return subclouds


def colorize_subcloud(scene, folder_path, foldername, subcloud_name):
    output_name = os.path.join(folder_path, foldername, "12_octree", subcloud_name + "_colorized.ply")
    if os.path.exists(output_name):
        return
    scene.parse_agisoft_xml(
        os.path.join(folder_path, foldername, subcloud_name + ".xml"))
    scene.cache_images(
        [
            os.path.join(folder_path, foldername, "1_crack/"),
            os.path.join(folder_path, foldername, "2_spall/"),
            os.path.join(folder_path, foldername, "3_corr/"),
            os.path.join(folder_path, foldername, "4_effl/"),
            os.path.join(folder_path, foldername, "5_vege/"),
            os.path.join(folder_path, foldername, "6_cp/"),
            os.path.join(folder_path, foldername, "8_background/")
        ],
        os.path.join(folder_path, foldername, "9_sharpness/"),
        os.path.join(folder_path, foldername, "10_depth/"),
        0.5
    )
    scene.colorize_point_cloud(
        os.path.join(folder_path, foldername, "12_octree", subcloud_name + ".ply"),
        output_name
    )


def recombine_subclouds(source_path, target_path):
    cloud = None

    # recombine point cloud
    for i, f in enumerate([f for f in os.listdir(source_path) if "colorized.ply" in f]):
        subcloud = PyntCloud.from_file(os.path.join(source_path, f))
        if i == 0:
            cloud = subcloud
        else:
            cloud.points = pd.concat([cloud.points, subcloud.points])

    cloud.to_file(target_path)
