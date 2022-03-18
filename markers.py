import os
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud
import xml.etree.ElementTree as ET


def load_agisoft_markers(path):
    """ Loads all markers from agisoft xml. """
    root = ET.parse(path).getroot()

    markers = {}
    for marker in root.iter('marker'):
        if 'marker_id' in marker.attrib.keys():
            continue
        print(marker.attrib['label'])
        idd = np.int32(marker.attrib['id'])
        print(idd)
        label = marker.attrib['label']
        ref = marker.find('reference')
        enabled = True if ref.attrib['enabled'] is None else False
        if True or enabled:
            markers[idd] = np.float32([ref.attrib['x'], ref.attrib['y'], ref.attrib['z']])

    return markers


def export_agisoft_markers(markers, path):
    """ Save marker to agisoft xml file. """
    doc = ET.Element("document")
    doc.tail = "\n"
    doc.text = "\n\t"
    chunk = ET.SubElement(doc, "chunk")
    chunk.tail = "\n"
    chunk.text = "\n\t\t"
    mark = ET.SubElement(chunk, "markers")
    mark.tail = "\n\t"
    mark.text = "\n\t\t\t"

    for key in markers.keys():
        m = ET.SubElement(mark, "marker")
        m.tail = "\n\t\t\t"
        m.text = "\n\t\t\t\t"
        m.attrib['id'] = str(key)
        ref = ET.SubElement(m, "reference")
        ref.tail = "\n\t\t\t"
        ref.attrib['x'] = f"{markers[key][0]:f}"
        ref.attrib['y'] = f"{markers[key][1]:f}"
        ref.attrib['z'] = f"{markers[key][2]:f}"

    xml = ET.tostring(doc, encoding='utf8', method='xml').decode("utf-8")
    print(xml)

    with open(path, 'w') as f:
        f.write(xml)


def extract_markers(ply_path):
    """ Extract markers from point cloud. """
    cloud = PyntCloud.from_file(ply_path)

    defect = np.array(cloud.points["defect"])
    cluster = np.array(cloud.points["cluster"])

    if hasattr(cloud.points, 'normal_x'):
        cloud.points["nx"] = cloud.points["normal_x"]
        cloud.points["ny"] = cloud.points["normal_y"]
        cloud.points["nz"] = cloud.points["normal_z"]

    cloud = cloud.to_instance("open3d", mesh=False, normals=True)
    cloud.paint_uniform_color([1.0, 0.0, 1.0])

    # only keep control points
    clusters = np.unique(cluster[defect == 1], return_counts=True)

    # loop over control points
    markers = {}
    for c in clusters[0]:
        idxs = np.nonzero(cluster == c)[0]
        subcloud = cloud.select_by_index(idxs)
        markers[c] = subcloud.get_center()

    return markers

def refine_markers(source_markers):
    # TODO
    pass

