"""Functions for working with bounding boxes.
By convention, a box is represented as the topleft x,y coordinates and the width and height:
[x1, y1, width, height].
"""

from boxlib.boxlib import area, bb_of_mask, bb_of_points, box_around, box_hull, center, contains, \
    corners, crop_to_square, expand, expand_to_square, full, empty, giou, intersection, \
    intersection_vertical, iou, random_partial_subbox, shift, side_midpoints, crop_image
