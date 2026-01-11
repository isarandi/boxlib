# boxlib

A lightweight Python library for bounding box manipulation, designed for computer vision tasks.

## Installation

```bash
pip install boxlib
```

## Box Convention

All bounding boxes are NumPy arrays with format `[x, y, width, height]`, where `(x, y)` is the top-left corner.

```python
import numpy as np
import boxlib

box = np.array([10, 20, 100, 50])  # x=10, y=20, width=100, height=50
```

## Quick Examples

```python
import numpy as np
import boxlib

box = np.array([10, 20, 100, 50])

# Get center point
boxlib.center(box)  # array([60., 45.])

# Expand box by 2x around its center
boxlib.expand(box, 2)  # array([-40., -5., 200., 100.])

# Make box square (using max dimension)
boxlib.expand_to_square(box)  # array([10., -5., 100., 100.])

# Intersection of two boxes
box2 = np.array([50, 40, 100, 50])
boxlib.intersection(box, box2)  # array([50., 40., 60., 30.])

# IoU (Intersection over Union)
boxlib.iou(box, box2)  # 0.189...

# Crop image to bounding box
image = np.zeros((200, 200, 3))
cropped = boxlib.crop_image(image, box)  # shape: (50, 100, 3)
```

## Functions

### Geometry
- `center(box)` - Get center point
- `corners(box)` - Get four corner coordinates
- `side_midpoints(box)` - Get midpoints of each side
- `inscribed_ellipse_points(box, n_angles, n_radii)` - Sample points inside inscribed ellipse
- `area(box)` - Calculate area

### Transformations
- `shift(box, delta)` - Translate box by offset
- `expand(box, factor)` - Scale box around its center
- `expand_to_square(box)` - Expand to square using max dimension
- `crop_to_square(box)` - Crop to square using min dimension
- `box_around(center, size)` - Create box from center point and size

### Set Operations
- `intersection(box1, box2)` - Intersection of two boxes
- `box_hull(box1, box2)` - Smallest box containing both
- `contains(box, points)` - Check if points are inside box

### Metrics
- `iou(box1, box2)` - Intersection over Union
- `giou(box1, box2)` - Generalized IoU

### Construction
- `bb_of_points(points)` - Bounding box of point set
- `bb_of_mask(mask)` - Bounding box of binary mask
- `full(imshape=None, imsize=None)` - Full image bounding box
- `empty()` - Zero-size box at origin

### Image Operations
- `crop_image(image, box, pad=True)` - Crop image to box with optional padding

## License

MIT