"""Functions for working with bounding boxes.
By convention, a box is represented as the topleft x,y coordinates and the width and height:
[x1, y1, width, height].
"""
import numpy as np


def expand(bbox, expansion_factor=1):
    center_point = center(bbox)
    new_size = bbox[2:] * expansion_factor
    return np.concatenate([center_point - new_size / 2, new_size])


def center(box):
    return box[:2] + box[2:4] / 2


def expand_to_square(box):
    center_point = center(box)
    side = np.max(box[2:4])
    return np.array([center_point[0] - side / 2, center_point[1] - side / 2, side, side])


def intersect(box, other_box):
    topleft = np.maximum(box[:2], other_box[:2])
    bottomright = np.minimum(box[:2] + box[2:4], other_box[:2] + other_box[2:4])
    return np.concatenate([topleft, np.maximum(0, bottomright - topleft)])


def box_hull(box, other_box):
    topleft = np.minimum(box[:2], other_box[:2])
    bottomright = np.maximum(box[:2] + box[2:4], other_box[:2] + other_box[2:4])
    return np.concatenate([topleft, bottomright - topleft])


def box_around(center_point, size):
    center_point = np.array(center_point)
    size = np.array(size)
    if size.size == 1:
        size = size.reshape(-1)[0]
        size = np.array([size, size])
    return np.concatenate([center_point - size / 2, size])


def corners(box):
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def iou(box1, box2):
    box1 = np.asarray(box1, np.float32)
    box2 = np.asarray(box2, np.float32)

    intersection_area = area(intersect(box1, box2))
    union_area = area(box1) + area(box2) - intersection_area
    return intersection_area / union_area


def contains(box, points):
    start = np.asarray(box[:2])
    end = np.asarray(box[:2] + box[2:4])
    points = np.asarray(points)
    return np.all(np.logical_and(start <= points, points < end), axis=-1)


def area(box):
    return box[2] * box[3]


def bb_of_points(points):
    if len(points) == 0:
        return np.zeros(4, np.float32)
    x1, y1 = np.nanmin(points, axis=0)
    x2, y2 = np.nanmax(points, axis=0)
    return np.asarray([x1, y1, x2 - x1, y2 - y1])


def full_box(imshape=None, imsize=None):
    assert imshape is not None or imsize is not None
    if imshape is None:
        imshape = [imsize[1], imsize[0]]
    return np.asarray([0, 0, imshape[1], imshape[0]])


def intersect_vertical(box, other_box):
    top = np.maximum(box[1], other_box[1])
    bottom = np.minimum(box[1] + box[3], other_box[1] + other_box[3])
    return np.array([box[0], top, box[2], bottom - top])


def random_partial_box(random_state):
    """Sample a square uniformly from inside the unit square, such that it has side length >= 0.5"""
    while True:
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if 0.5 < side < y2:
            return np.array([x1, y2 - side, side, side])


def random_partial_subbox(box, random_state):
    subbox = random_partial_box(random_state)
    topleft = box[:2] + subbox[:2] * box[2:]
    size = subbox[2:] * box[2:]
    return np.concatenate([topleft, size])
