boxlib
======

A lightweight Python library for bounding box manipulation, designed for computer vision tasks.

Installation
------------

.. code-block:: bash

   pip install boxlib

Box Convention
--------------

All bounding boxes are NumPy arrays with format ``[x, y, width, height]``, where ``(x, y)`` is the top-left corner.

.. code-block:: python

   import numpy as np
   import boxlib

   box = np.array([10, 20, 100, 50])  # x=10, y=20, width=100, height=50

Quick Examples
--------------

.. code-block:: python

   import numpy as np
   import boxlib

   box = np.array([10, 20, 100, 50])

   # Get center point
   boxlib.center(box)  # array([60., 45.])

   # Expand box by 2x around its center
   boxlib.expand(box, 2)

   # Make box square (using max dimension)
   boxlib.expand_to_square(box)

   # Intersection of two boxes
   box2 = np.array([50, 40, 100, 50])
   boxlib.intersection(box, box2)

   # IoU (Intersection over Union)
   boxlib.iou(box, box2)

   # Crop image to bounding box
   image = np.zeros((200, 200, 3))
   cropped = boxlib.crop_image(image, box)

Full API Reference
------------------

All functions are documented in the :doc:`API Reference <api/boxlib/index>`.

.. toctree::
   :maxdepth: 2
   :hidden:

   API Reference <api/boxlib/index>