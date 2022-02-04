import os
import math
import open3d as o3d

from pyntcloud import PyntCloud


def store_subcloud(node, node_info, scene, cloud, path, count):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode) and isinstance(node,
                                                                        o3d.geometry.OctreeInternalPointNode) and len(
        node.indices) < count \
            or isinstance(node, o3d.geometry.OctreeLeafNode) and isinstance(node,
                                                                            o3d.geometry.OctreePointColorLeafNode):
        subcloud = cloud.select_by_index(node.indices)
        o3d.io.write_point_cloud(
            os.path.join(path, f"subcloud_{node_info.depth}_{node_info.child_index}_{len(node.indices)}.ply"), subcloud)
        early_stop = True

        scene.parse_agisoft_xml(os.path.join(path, "cameras.xml"))
        scene.filter_cameras(os.path.join(path, "12_octree",
                                          f"subcloud_{node_info.depth}_{node_info.child_index}_{len(node.indices)}.ply"),
                             os.path.join(path, "12_octree",
                                          f"subcloud_{node_info.depth}_{node_info.child_index}_{len(node.indices)}.xml"))
    return early_stop


def create_octree(scene, folder_path, foldername, cloudname):
    cloud = o3d.io.read_point_cloud(os.path.join(folder_path, foldername, cloudname))

    # estimate required depth
    count = 100000
    depth = int(math.log(len(cloud.points) / count) / math.log(8)) + 2

    # create octree
    octree = o3d.geometry.Octree(max_depth=depth)
    octree.convert_from_point_cloud(cloud, size_expand=0.00)
    octree.traverse(
        lambda node, node_info: store_subcloud(node, node_info, scene, folder_path, foldername, cloudname, count))


def colorize_subcloud(scene, folder_path, foldername, subcloud_name):
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
        os.path.join(folder_path, foldername, "12_octree", subcloud_name + "_colorized.ply")
    )


def recombine_subclouds(folder_path, foldername, cloudname):
    path = os.path.join(folder_path, foldername, "12_octree")

    cloud = None

    # recombine point cloud
    for i, f in enumerate([f for f in os.listdir(path) if "colorized.ply" in f]):
        subcloud = PyntCloud.from_file(os.path.join(path, f))
        if i == 0:
            cloud = subcloud
        else:
            cloud.points = cloud.points.append(subcloud.points)

    cloud.to_file(os.path.join(folder_path, foldername, cloudname.replace(".", "_colorized2.")))
