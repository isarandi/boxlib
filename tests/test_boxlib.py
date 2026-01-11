import numpy as np
import numpy.testing as npt
import pytest

import boxlib


class TestCenter:
    def test_basic(self):
        box = np.array([10, 20, 100, 50])
        npt.assert_array_equal(boxlib.center(box), [60, 45])

    def test_zero_size(self):
        box = np.array([10, 20, 0, 0])
        npt.assert_array_equal(boxlib.center(box), [10, 20])


class TestBoxAround:
    def test_basic(self):
        result = boxlib.box_around([50, 50], [20, 30])
        npt.assert_array_equal(result, [40, 35, 20, 30])

    def test_scalar_size(self):
        result = boxlib.box_around([50, 50], 20)
        npt.assert_array_equal(result, [40, 40, 20, 20])


class TestExpand:
    def test_expand_double(self):
        box = np.array([10, 10, 20, 20])
        result = boxlib.expand(box, 2)
        npt.assert_array_equal(result, [0, 0, 40, 40])

    def test_expand_half(self):
        box = np.array([0, 0, 100, 100])
        result = boxlib.expand(box, 0.5)
        npt.assert_array_equal(result, [25, 25, 50, 50])

    def test_no_expansion(self):
        box = np.array([10, 20, 30, 40])
        result = boxlib.expand(box, 1)
        npt.assert_array_equal(result, box)


class TestExpandToSquare:
    def test_wider_box(self):
        box = np.array([0, 0, 100, 50])
        result = boxlib.expand_to_square(box)
        # Wider box: pad in y direction
        npt.assert_array_equal(result, [0, -25, 100, 100])

    def test_taller_box(self):
        box = np.array([0, 0, 50, 100])
        result = boxlib.expand_to_square(box)
        # Taller box: pad in x direction
        npt.assert_array_equal(result, [-25, 0, 100, 100])

    def test_already_square(self):
        box = np.array([10, 20, 50, 50])
        result = boxlib.expand_to_square(box)
        npt.assert_array_equal(result, [10, 20, 50, 50])


class TestCropToSquare:
    def test_wider_box(self):
        box = np.array([0, 0, 100, 50])
        result = boxlib.crop_to_square(box)
        npt.assert_array_equal(result, [25, 0, 50, 50])

    def test_taller_box(self):
        box = np.array([0, 0, 50, 100])
        result = boxlib.crop_to_square(box)
        npt.assert_array_equal(result, [0, 25, 50, 50])


class TestIntersection:
    def test_overlapping(self):
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 100, 100])
        result = boxlib.intersection(box1, box2)
        npt.assert_array_equal(result, [50, 50, 50, 50])

    def test_no_overlap(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 10, 10])
        result = boxlib.intersection(box1, box2)
        assert result[2] == 0 or result[3] == 0  # zero area

    def test_contained(self):
        outer = np.array([0, 0, 100, 100])
        inner = np.array([25, 25, 50, 50])
        result = boxlib.intersection(outer, inner)
        npt.assert_array_equal(result, inner)


class TestBoxHull:
    def test_basic(self):
        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([25, 25, 50, 50])
        result = boxlib.box_hull(box1, box2)
        npt.assert_array_equal(result, [0, 0, 75, 75])

    def test_disjoint(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 10, 10])
        result = boxlib.box_hull(box1, box2)
        npt.assert_array_equal(result, [0, 0, 30, 30])


class TestCorners:
    def test_basic(self):
        box = np.array([10, 20, 30, 40])
        result = boxlib.corners(box)
        expected = np.array([[10, 20], [40, 20], [40, 60], [10, 60]], np.float32)
        npt.assert_array_equal(result, expected)


class TestSideMidpoints:
    def test_basic(self):
        box = np.array([0, 0, 100, 100])
        result = boxlib.side_midpoints(box)
        expected = np.array([[0, 50], [50, 0], [100, 50], [50, 100]], np.float32)
        npt.assert_array_equal(result, expected)


class TestIou:
    def test_identical(self):
        box = np.array([0, 0, 100, 100])
        assert boxlib.iou(box, box) == 1.0

    def test_no_overlap(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 10, 10])
        assert boxlib.iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 0, 100, 100])
        # intersection: 50x100=5000, union: 10000+10000-5000=15000
        assert boxlib.iou(box1, box2) == pytest.approx(5000 / 15000)


class TestGiou:
    def test_identical(self):
        box = np.array([0, 0, 100, 100])
        assert boxlib.giou(box, box) == 1.0

    def test_no_overlap_same_size(self):
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([10, 0, 10, 10])
        # IoU=0, union=200, hull=200, so GIoU = 0 + 200/200 - 1 = 0
        assert boxlib.giou(box1, box2) == pytest.approx(0.0)


class TestContains:
    def test_inside(self):
        box = np.array([0, 0, 100, 100])
        points = np.array([[50, 50], [0, 0], [99, 99]])
        result = boxlib.contains(box, points)
        npt.assert_array_equal(result, [True, True, True])

    def test_outside(self):
        box = np.array([0, 0, 100, 100])
        points = np.array([[100, 100], [-1, 50], [50, 100]])
        result = boxlib.contains(box, points)
        npt.assert_array_equal(result, [False, False, False])


class TestArea:
    def test_basic(self):
        box = np.array([0, 0, 10, 20])
        assert boxlib.area(box) == 200

    def test_zero(self):
        box = np.array([10, 20, 0, 0])
        assert boxlib.area(box) == 0


class TestBbOfPoints:
    def test_basic(self):
        points = np.array([[10, 20], [30, 40], [20, 30]])
        result = boxlib.bb_of_points(points)
        npt.assert_array_equal(result, [10, 20, 20, 20])

    def test_empty(self):
        points = np.array([]).reshape(0, 2)
        result = boxlib.bb_of_points(points)
        npt.assert_array_equal(result, [0, 0, 0, 0])

    def test_single_point(self):
        points = np.array([[5, 10]])
        result = boxlib.bb_of_points(points)
        npt.assert_array_equal(result, [5, 10, 0, 0])


class TestFull:
    def test_from_imshape(self):
        result = boxlib.full(imshape=(480, 640))
        npt.assert_array_equal(result, [0, 0, 640, 480])

    def test_from_imsize(self):
        result = boxlib.full(imsize=(640, 480))
        npt.assert_array_equal(result, [0, 0, 640, 480])


class TestEmpty:
    def test_basic(self):
        result = boxlib.empty()
        npt.assert_array_equal(result, [0, 0, 0, 0])


class TestShift:
    def test_basic(self):
        box = np.array([10, 20, 30, 40])
        result = boxlib.shift(box, [5, -5])
        npt.assert_array_equal(result, [15, 15, 30, 40])


class TestBbOfMask:
    def test_basic(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True
        result = boxlib.bb_of_mask(mask)
        npt.assert_array_equal(result, [30, 20, 30, 20])

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        result = boxlib.bb_of_mask(mask)
        npt.assert_array_equal(result, [0, 0, 0, 0])


class TestCropImage:
    def test_basic(self):
        image = np.arange(100).reshape(10, 10, 1)
        box = np.array([2, 3, 4, 5])
        result = boxlib.crop_image(image, box)
        assert result.shape == (5, 4, 1)

    def test_with_padding(self):
        image = np.ones((10, 10, 3))
        box = np.array([-2, -2, 6, 6])
        result = boxlib.crop_image(image, box, pad=True, pad_value=0)
        assert result.shape == (6, 6, 3)
        assert result[0, 0, 0] == 0  # padded region
        assert result[3, 3, 0] == 1  # original region

    def test_no_padding(self):
        # pad=False clips start coords to 0 but keeps requested size if available
        image = np.ones((10, 10, 3))
        box = np.array([-2, -2, 6, 6])
        result = boxlib.crop_image(image, box, pad=False)
        assert result.shape == (6, 6, 3)


class TestRandomPartialSubbox:
    def test_within_bounds(self):
        rng = np.random.RandomState(42)
        box = np.array([100, 200, 50, 80])
        for _ in range(100):
            subbox = boxlib.random_partial_subbox(box, rng)
            # subbox should be within the original box
            assert subbox[0] >= box[0]
            assert subbox[1] >= box[1]
            assert subbox[0] + subbox[2] <= box[0] + box[2]
            assert subbox[1] + subbox[3] <= box[1] + box[3]