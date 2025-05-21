import warnings

import numpy as np


def center(box: np.ndarray) -> np.ndarray:
    """Get the center point of a bounding box.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The center point of the bounding box as a numpy array of shape (2,).
    """
    return box[:2] + box[2:4] / 2


def box_around(center_point: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Create a bounding box around a center point, with given size.

    Args:
        center_point: The center point of the bounding box.
        size: The size of the bounding box. Can be a single number or a numpy array of shape (2,).

    Returns:
        The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
    """
    center_point = np.array(center_point)
    size = np.array(size)
    if size.size == 1:
        size = size.reshape(-1)[0]
        size = np.array([size, size])
    return np.concatenate([center_point - size / 2, size])


def expand(bbox: np.ndarray, expansion_factor: float = 1) -> np.ndarray:
    """Expand a bounding box around its center, by a given factor.

    Args:
        bbox: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        expansion_factor: The factor to expand the bounding box by.

    Returns:
        The expanded bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    """
    new_size = bbox[2:4] * expansion_factor
    return box_around(center(bbox), new_size)


def expand_to_square(box: np.ndarray) -> np.ndarray:
    """Expand a bounding box to a square, by taking the maximum of the width and height.

    Padding is added symmetrically for the shorter side.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The expanded bounding box as a numpy array of shape (4,), [x1, y1, M, M], \
            where M is the maximum of the width and height.
    """

    return box_around(center(box), np.max(box[2:4]))


def crop_to_square(box):
    """Crop a bounding box to a square, by taking the minimum of the width and height.

    Pixels are cropped away symmetrically from the longer side.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The cropped bounding box as a numpy array of shape (4,), [x1, y1, m, m], \
            where m is the minimum of the width and height.
    """
    return box_around(center(box), np.min(box[2:4]))


def intersection(box: np.ndarray, other_box: np.ndarray) -> np.ndarray:
    """Get the intersection of two bounding boxes.

    Args:
        box: The first bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        other_box: The second bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The intersection of the two bounding boxes as a numpy array of \
            shape (4,), [x1, y1, width, height]. If the boxes do not intersect, \
            the returned box will have zero width and height.
    """
    topleft = np.maximum(box[:2], other_box[:2])
    bottomright = np.minimum(box[:2] + box[2:4], other_box[:2] + other_box[2:4])
    return np.concatenate([topleft, np.maximum(0, bottomright - topleft)])


def box_hull(box: np.ndarray, other_box: np.ndarray) -> np.ndarray:
    """Get the smallest bounding box that contains both input bounding boxes.

    Args:
        box: The first bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        other_box: The second bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The smallest bounding box that contains both input bounding boxes as a numpy array of \
            shape (4,), [x1, y1, width, height].
    """
    topleft = np.minimum(box[:2], other_box[:2])
    bottomright = np.maximum(box[:2] + box[2:4], other_box[:2] + other_box[2:4])
    return np.concatenate([topleft, bottomright - topleft])


def corners(box: np.ndarray) -> np.ndarray:
    """Get the coordinates of the four corners of a bounding box.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The coordinates of the four corners of the bounding box as a numpy array of shape (4, 2), \
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].
    """

    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def side_midpoints(box: np.ndarray) -> np.ndarray:
    """Get the coordinates of the midpoints of the four sides of a bounding box.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The coordinates of the midpoints of the four sides of the bounding box as a numpy array of
        shape (4, 2).
    """

    x, y, w, h = box
    return np.array(
        [[x, y + h / 2], [x + w / 2, y], [x + w, y + h / 2], [x + w / 2, y + h]],
        np.float32,
    )


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: The first bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        box2: The second bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The Intersection over Union (IoU) of the two bounding boxes as a float.
    """
    box1 = np.asarray(box1, np.float32)
    box2 = np.asarray(box2, np.float32)

    intersection_area = area(intersection(box1, box2))
    union_area = area(box1) + area(box2) - intersection_area
    return intersection_area / union_area


def giou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate the Generalized Intersection over Union (IoU) of two bounding boxes.

    The generalized IoU is defined as:

    GIoU = IoU + U / C - 1,

    where C is the area of smallest bounding box that contains both input bounding boxes, and U is
    the area of the union of the two bounding boxes.

    The advantage of GIoU over IoU is that it can measure a continuous similarity of two bounding
    boxes even if they do not intersect.

    Args:
        box1: The first bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        box2: The second bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The Generalized Intersection over Union (IoU) of the two bounding boxes as a float.
    """

    box1 = np.asarray(box1, np.float32)
    box2 = np.asarray(box2, np.float32)
    full_box = box_hull(box1, box2)
    intersection_area = area(intersection(box1, box2))
    union_area = area(box1) + area(box2) - intersection_area
    return intersection_area / union_area + union_area / area(full_box) - 1


def contains(box: np.ndarray, points: np.ndarray) -> np.ndarray[bool]:
    """Check if a set of points are contained within a bounding box.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        points: The points to check as a numpy array of shape (N, 2), [[x1, y1], [x2, y2], ...].

    Returns:
        A boolean numpy array of shape (N,), indicating if each point is contained within the \
            bounding box.
    """
    start = np.asarray(box[:2])
    end = np.asarray(box[:2] + box[2:4])
    points = np.asarray(points)
    return np.all(np.logical_and(start <= points, points < end), axis=-1)


def area(box: np.ndarray) -> float:
    """Calculate the area of a bounding box.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].

    Returns:
        The area of the bounding box as a float.
    """
    return box[2] * box[3]


def bb_of_points(points: np.ndarray) -> np.ndarray:
    """Construct the smmallest bounding box that contains a set of points.

    Args:
        points: The points as a numpy array of shape (N, 2), [[x1, y1], [x2, y2], ...].

    Returns:
        The smallest bounding box that contains the points as a numpy array of shape (4,), \
            [x1, y1, width, height]. If the input is empty, the returned box will have zero width \
            and height.
    """

    if len(points) == 0:
        return np.zeros(4, np.float32)

    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            x1, y1 = np.nanmin(points, axis=0)
            x2, y2 = np.nanmax(points, axis=0)

    result = np.asarray([x1, y1, x2 - x1, y2 - y1])
    if np.any(np.isnan(result)):
        return np.zeros(4, np.float32)
    return result


def full(imshape=None, imsize=None) -> np.ndarray:
    """Create a bounding box that covers the full image.

    Args:
        imshape: The shape of the image as a tuple of (height, width).
        imsize: The size of the image as a tuple of (width, height).

    Returns:
        The bounding box that covers the full image as a numpy array of shape (4,), \
            [x1, y1, width, height].

    Note:
        Exactly one of ``imshape`` or ``imsize`` must be provided.
    """
    assert imshape is not None or imsize is not None
    if imshape is None:
        imshape = [imsize[1], imsize[0]]
    return np.asarray([0, 0, imshape[1], imshape[0]])


def empty() -> np.ndarray:
    """Create an empty bounding box.

    Returns:
        An empty bounding box as a numpy array of shape (4,), [0, 0, 0, 0].
    """
    return np.array([0, 0, 0, 0], np.float32)


def intersection_vertical(box, other_box):
    top = np.maximum(box[1], other_box[1])
    bottom = np.minimum(box[1] + box[3], other_box[1] + other_box[3])
    return np.array([box[0], top, box[2], bottom - top])


def random_partial_box(random_state: np.random.RandomState) -> np.ndarray:
    """Sample a square uniformly from inside the unit square, such that it has side length >= 0.5

    Args:
        random_state: RandomState object for reproducibility.

    Returns:
        The sampled square as a numpy array of shape (4,), [x1, y1, width, height].
    """

    while True:
        x1 = random_state.uniform(0, 0.5)
        x2, y2 = random_state.uniform(0.5, 1, size=2)
        side = x2 - x1
        if 0.5 < side < y2:
            return np.array([x1, y2 - side, side, side])


def random_partial_subbox(box: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
    """Sample a subbox uniformly from inside the given box, with the same aspect ratio.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        random_state: RandomState object for reproducibility.

    Returns:
        The sampled subbox as a numpy array of shape (4,), [x1, y1, width, height].
    """
    subbox = random_partial_box(random_state)
    topleft = box[:2] + subbox[:2] * box[2:4]
    size = subbox[2:] * box[2:4]
    return np.concatenate([topleft, size])


def shift(box: np.ndarray, delta) -> np.ndarray:
    """Shift a bounding box by a given amount.

    Args:
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        delta: The amount to shift the bounding box by as a numpy array of shape (2,), [dx, dy], or
            a single number to shift by in both x and y directions.

    Returns:
        The shifted bounding box.
    """
    return np.concatenate([box[:2] + delta, box[2:4]])


def bb_of_mask(mask: np.ndarray) -> np.ndarray:
    """Get the bounding box of a binary mask.

    Args:
        mask: The binary mask as a numpy array of shape (H, W).

    Returns:
        The bounding box of the mask as a numpy array of shape (4,), [x1, y1, width, height].
    """

    try:
        xmin, xmax = np.nonzero(np.any(mask, axis=0))[0][[0, -1]]
        ymin, ymax = np.nonzero(np.any(mask, axis=1))[0][[0, -1]]
        return np.array([xmin, ymin, xmax - xmin + 1, ymax - ymin + 1], np.float32)
    except IndexError:
        return empty()


def crop_image(image: np.ndarray, box: np.ndarray, pad: bool = True, pad_value=0) -> np.ndarray:
    """Crop (and possibly pad) an image to a bounding box.

    Args:
        image: The image as a numpy array of shape (H, W, C).
        box: The bounding box as a numpy array of shape (4,), [x1, y1, width, height].
        pad: Whether to pad the image if the box goes outside the image boundaries.
        pad_value: The value to pad the image with.

    Returns:
        The cropped (and possibly padded) image.
    """
    intbox = np.round(np.asarray(box)).astype(np.int32)

    # check if start is negative or the end goes beyond the imshape. If so, then pad
    paddings = [[0, 0], [0, 0], [0, 0]]
    if pad:
        if intbox[0] < 0:
            paddings[1][0] = np.abs(intbox[0])
            intbox[2] += intbox[0]
            intbox[0] = 0
        if intbox[1] < 0:
            paddings[0][0] = np.abs(intbox[1])
            intbox[3] += intbox[1]
            intbox[1] = 0
        if intbox[0] + intbox[2] > image.shape[1]:
            paddings[1][1] = intbox[0] + intbox[2] - image.shape[1]
            intbox[2] = image.shape[1] - intbox[0]
        if intbox[1] + intbox[3] > image.shape[0]:
            paddings[0][1] = intbox[1] + intbox[3] - image.shape[0]
            intbox[3] = image.shape[0] - intbox[1]
    else:
        intbox[0] = np.maximum(0, intbox[0])
        intbox[1] = np.maximum(0, intbox[1])
        intbox[2] = np.minimum(image.shape[1] - intbox[0], intbox[2])
        intbox[3] = np.minimum(image.shape[0] - intbox[1], intbox[3])

    cropped = image[intbox[1] : intbox[1] + intbox[3], intbox[0] : intbox[0] + intbox[2]]
    if pad:
        cropped = np.pad(cropped, paddings, mode="constant", constant_values=pad_value)
    return cropped
