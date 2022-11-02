#!/usr/bin/env python3

from functools import lru_cache

from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import builtin
from megengine.core.tensor.utils import subgraph

__all__ = ["box_iou", "box_ioa", "box_center", "point_distance"]


def _subtensor(f, src, axis, begin, end, step, idx):
    items = [
        [axis, begin is not None, end is not None, step is not None, idx is not None]
    ]
    args = []
    if begin is not None:
        args += (begin,)
    if end is not None:
        args += (end,)
    if step is not None:
        args += (step,)
    if idx is not None:
        args += (idx,)
    return f(builtin.Subtensor(items=items), src, *args)


_Elemwise = builtin.Elemwise
_ElemMode = builtin.Elemwise.Mode


@lru_cache(maxsize=None)
def _get_iou_func(device, dtype, gopt_level=2):
    @subgraph("IOU", dtype, device, 2, gopt_level=gopt_level)
    def IOU(inputs, f, c):
        boxes1, boxes2 = inputs
        boxes1 = f(builtin.AddAxis(axis=[1]), boxes1)
        boxes2 = f(builtin.AddAxis(axis=[0]), boxes2)

        boxes1_0 = _subtensor(f, boxes1, 2, None, None, None, c(0, dtype="int32"))
        boxes1_1 = _subtensor(f, boxes1, 2, None, None, None, c(1, dtype="int32"))
        boxes1_2 = _subtensor(f, boxes1, 2, None, None, None, c(2, dtype="int32"))
        boxes1_3 = _subtensor(f, boxes1, 2, None, None, None, c(3, dtype="int32"))

        boxes2_0 = _subtensor(f, boxes2, 2, None, None, None, c(0, dtype="int32"))
        boxes2_1 = _subtensor(f, boxes2, 2, None, None, None, c(1, dtype="int32"))
        boxes2_2 = _subtensor(f, boxes2, 2, None, None, None, c(2, dtype="int32"))
        boxes2_3 = _subtensor(f, boxes2, 2, None, None, None, c(3, dtype="int32"))

        iw_lhs = f(_Elemwise(mode=_ElemMode.MIN), boxes1_2, boxes2_2)
        iw_rhs = f(_Elemwise(mode=_ElemMode.MAX), boxes1_0, boxes2_0)
        iw = f(_Elemwise(mode=_ElemMode.SUB), iw_lhs, iw_rhs)

        ih_lhs = f(_Elemwise(mode=_ElemMode.MIN), boxes1_3, boxes2_3)
        ih_rhs = f("max", boxes1_1, boxes2_1)
        ih = f("-", ih_lhs, ih_rhs)

        iw = f(_Elemwise(mode=_ElemMode.MAX), iw, c(0, dtype="float32"))
        ih = f(_Elemwise(mode=_ElemMode.MAX), ih, c(0, dtype="float32"))
        inter = f(_Elemwise(mode=_ElemMode.MUL), iw, ih)

        boxes1_width = f(_Elemwise(mode=_ElemMode.SUB), boxes1_2, boxes1_0)
        boxes1_height = f(_Elemwise(mode=_ElemMode.SUB), boxes1_3, boxes1_1)
        boxes1_area = f(_Elemwise(mode=_ElemMode.MUL), boxes1_width, boxes1_height,)

        boxes2_width = f(_Elemwise(mode=_ElemMode.SUB), boxes2_2, boxes2_0)
        boxes2_height = f(_Elemwise(mode=_ElemMode.SUB), boxes2_3, boxes2_1)
        boxes2_area = f(_Elemwise(mode=_ElemMode.MUL), boxes2_width, boxes2_height,)

        union = f(_Elemwise(mode=_ElemMode.ADD), boxes1_area, boxes2_area)
        union = f(_Elemwise(mode=_ElemMode.SUB), union, inter)

        iou = f(_Elemwise(mode=_ElemMode.TRUE_DIV), inter, union)
        iou = f(_Elemwise(mode=_ElemMode.MAX), iou, c(0, dtype="float32"))

        return [iou], [False]

    return IOU


def box_iou(boxes1, boxes2):
    r"""Computing the intersection-over-union between two sets of bounding boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1: first set of boxes with shape ``(m, 4)``
        boxes2: second set of boxes with shape ``(n, 4)``

    Returns:
        output tensor with shape ``(m, n)``, the m*n matrix containing the pairwise
        IoU values for every element in boxes1 and boxes2
    """

    IOU = _get_iou_func(boxes1.device, boxes1.dtype)
    return apply(IOU(), boxes1, boxes2)[0]


@lru_cache(maxsize=None)
def _get_Center(device, dtype, gopt_level=2):
    @subgraph("Center", dtype, device, 1, gopt_level=gopt_level)
    def _Center(inputs, f, c):
        boxes = inputs[0]

        top_left = _subtensor(f, boxes, 1, None, c(2, dtype="int32"), None, None)
        bottom_right = _subtensor(f, boxes, 1, c(-2, dtype="int32"), None, None, None)
        center = f(_Elemwise(mode=_ElemMode.ADD), top_left, bottom_right)
        center = f(_Elemwise(mode=_ElemMode.TRUE_DIV), center, c(2, dtype="float32"),)

        return [center], [False]

    return _Center


def box_center(boxes):
    r"""Computing the center of bounding boxes.

    Boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes: a set of boxes with shape ``(m, 4)``

    Returns:
        output tensor with shape ``(m, 2)``, the m*2 matrix containing the center
        of each bounding boxes
    """
    _Center = _get_Center(boxes.device, boxes.dtype)
    return apply(_Center(), boxes)[0]


@lru_cache(maxsize=None)
def _get_Distance(device, dtype, gopt_level=2):
    @subgraph("Distance", dtype, device, 2, gopt_level=gopt_level)
    def _Distance(inputs, f, c):
        lhs, rhs = inputs
        lhs = f(builtin.AddAxis(axis=[1]), lhs)
        diff = f(_Elemwise(mode=_ElemMode.SUB), lhs, rhs)
        diff_square = f("pow", diff, c(2, dtype="float32"),)
        square_sum = f(builtin.Reduce(mode="sum", axis=2), diff_square)
        square_sum = f(builtin.RemoveAxis(axis=[2]), square_sum)
        distance = f(
            _Elemwise(mode=_ElemMode.POW), square_sum, c(0.5, dtype="float32"),
        )

        return [distance], [False]

    return _Distance


def point_distance(points1, points2):
    """Computing the distance between two sets of points.

    Both sets of points are expected to be in ``(x, y)`` format.

    Args:
        points1: first set of boxes with shape ``(m, 2)``
        points2: second set of boxes with shape ``(n, 2)``

    Returns:
        output tensor with shape ``(m, n)``, the m*n matrix containing the pairwise
        distance for every element in points1 and points2
    """
    _Distance = _get_Distance(points1.device, points1.dtype)
    return apply(_Distance(), points1, points2)[0]


@lru_cache(maxsize=None)
def _get_ioa_func(device, dtype, gopt_level=2):
    @subgraph("IOA", dtype, device, 2, gopt_level=gopt_level)
    def IOA(inputs, f, c):
        boxes1, boxes2 = inputs
        boxes1 = f(builtin.AddAxis(axis=[1]), boxes1)
        boxes2 = f(builtin.AddAxis(axis=[0]), boxes2)

        boxes1_0 = _subtensor(f, boxes1, 2, None, None, None, c(0, dtype="int32"))
        boxes1_1 = _subtensor(f, boxes1, 2, None, None, None, c(1, dtype="int32"))
        boxes1_2 = _subtensor(f, boxes1, 2, None, None, None, c(2, dtype="int32"))
        boxes1_3 = _subtensor(f, boxes1, 2, None, None, None, c(3, dtype="int32"))

        boxes2_0 = _subtensor(f, boxes2, 2, None, None, None, c(0, dtype="int32"))
        boxes2_1 = _subtensor(f, boxes2, 2, None, None, None, c(1, dtype="int32"))
        boxes2_2 = _subtensor(f, boxes2, 2, None, None, None, c(2, dtype="int32"))
        boxes2_3 = _subtensor(f, boxes2, 2, None, None, None, c(3, dtype="int32"))

        iw_lhs = f(_Elemwise(mode=_ElemMode.MIN), boxes1_2, boxes2_2)
        iw_rhs = f(_Elemwise(mode=_ElemMode.MAX), boxes1_0, boxes2_0)
        iw = f(_Elemwise(mode=_ElemMode.SUB), iw_lhs, iw_rhs)

        ih_lhs = f(_Elemwise(mode=_ElemMode.MIN), boxes1_3, boxes2_3)
        ih_rhs = f("max", boxes1_1, boxes2_1)
        ih = f("-", ih_lhs, ih_rhs)

        iw = f(_Elemwise(mode=_ElemMode.MAX), iw, c(0, dtype="float32"))
        ih = f(_Elemwise(mode=_ElemMode.MAX), ih, c(0, dtype="float32"))
        inter = f(_Elemwise(mode=_ElemMode.MUL), iw, ih)

        boxes2_width = f(_Elemwise(mode=_ElemMode.SUB), boxes2_2, boxes2_0)
        boxes2_height = f(_Elemwise(mode=_ElemMode.SUB), boxes2_3, boxes2_1)
        boxes2_area = f(_Elemwise(mode=_ElemMode.MUL), boxes2_width, boxes2_height,)

        ioa = f(_Elemwise(mode=_ElemMode.TRUE_DIV), inter, boxes2_area)
        ioa = f(_Elemwise(mode=_ElemMode.MAX), ioa, c(0, dtype="float32"))

        return [ioa], [False]

    return IOA


def box_ioa(boxes1, boxes2):
    r"""Computing the intersection-over-area between two sets of bounding boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1: first set of boxes with shape ``(m, 4)``
        boxes2: second set of boxes with shape ``(n, 4)``

    Returns:
        output tensor with shape ``(m, n)``, the m*n matrix containing the pairwise
        IoA values for every element in boxes1 and boxes2
    """

    IOA = _get_ioa_func(boxes1.device, boxes1.dtype)
    return apply(IOA(), boxes1, boxes2)[0]
