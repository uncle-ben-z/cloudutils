import os
import cv2
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


def compute_linepoints(line, shape):
    """ Computes the intersection points of a homogeneous line with the frame of an image."""
    pleft = np.cross(line, [1, 0, 0])
    pleft /= pleft[-1]
    ptop = np.cross(line, [0, 1, 0])
    ptop /= ptop[-1]
    pright = np.cross(line, [1, 0, -shape[1]])
    pright /= pright[-1]
    pbottom = np.cross(line, [0, 1, -shape[0]])
    pbottom /= pbottom[-1]

    pts = np.empty((0, 3))
    if not np.any((pleft < 0) + (shape[1] < pleft)):
        pts = np.append(pts, pleft.reshape(1, 3), axis=0)
    if not np.any((ptop < 0) + (shape[0] < ptop)):
        pts = np.append(pts, ptop.reshape(1, 3), axis=0)
    if not np.any((pright < 0) + (shape[1] < pright)):
        pts = np.append(pts, pright.reshape(1, 3), axis=0)
    if not np.any((pbottom < 0) + (shape[0] < pbottom)):
        pts = np.append(pts, pbottom.reshape(1, 3), axis=0)

    return pts[:, :2]


def refine_markers(patch, offset):
    """ Adjusts the center of a control point based on the intersection of Hough lines in the image. """
    # hough transform
    edges = cv2.Canny(patch, 150, 200, apertureSize=3)
    # plt.imshow(patch)
    # plt.show()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 2)

    # get first line
    rho1, theta1 = lines[0][0]
    rho1 += np.cos(theta1) * offset[0] + np.sin(theta1) * offset[1]
    line1 = np.array(
        [np.cos(theta1), np.sin(theta1), -rho1])
    line1 /= line1[-1]

    # get second line
    for l in lines[1:, ...]:
        rho, theta = l[0]
        rho += np.cos(theta) * offset[0] + np.sin(theta) * offset[1]
        # apply angular deviation constraint
        dev = np.degrees(abs(theta1 - theta) % np.pi)
        if dev < 45:
            continue

        # compute intersection
        line2 = np.array([np.cos(theta), np.sin(theta), -rho])
        inter = np.cross(line1, line2)
        inter /= inter[-1]
        break

    return inter[:2], line1, line2


def refine_markers_harris(patch, offset):
    """ Heuristically uses the max Harris response for control point center. """
    harris = cv2.cornerHarris(patch, 2, 5, 0.07)
    edges = np.where(harris < 0, np.abs(harris), 0)

    point = np.array(np.where(harris == harris.max())).flatten()
    point += offset
    return np.float64(point)
