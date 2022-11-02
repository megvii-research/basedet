#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from enum import IntEnum, unique
import numpy as np

import megengine.functional as F
from megengine import tensor


@unique
class BoxMode(IntEnum):
    r"""
    Enum of different ways to represent a box tensor.
    Currently support two types of boxes, represented below.

    Single box represented like this (O represents origin):

    .. code:: none

        O--------------------------> x axis of an image
        |
        |     (x1, y1)
        |   ._____________________
        |   |                     |
        |  h|        . (x_c, y_c) |
        |   |_____________________.(x2, y2)
        |              w
        |
        v  y axis of an image

    """
    XYXY = 0
    """
    XYXY, (x0, y0, x1, y1) range in [0, width or height]. (x0, y0) is
    top-left point while (x1, y1) is bottom-right point.
    """
    XYWH = 1
    """
    XYWH, represented in (x0, y0, w, h) format. (w, h) means (width, height).
    """
    XcYcWH = 2
    """
    XcYcWH, represented in (x_c, y_c, w, h) format. (w, h) means (width, height).
    (x_c, y_c) is center point of boundding box.
    """


class BoxConverter:

    @classmethod
    def convert(cls, boxes, mode="xywh2xyxy"):
        # TODO: support multi-dim boxes
        # NOTE: only support 3 types of boxes.
        from_mode, to_mode = cls.get_from_mode_and_to_mode(mode)
        if from_mode == to_mode:
            return boxes

        # All mode boxes to XYWH format first
        if from_mode == BoxMode.XYWH:
            pass
        elif from_mode == BoxMode.XYXY:
            w, h = boxes[:, 2::4] - boxes[:, 0::4], boxes[:, 3::4] - boxes[:, 1::4]
            boxes = F.concat((boxes[:, :2], w, h), axis=1)
        elif from_mode == BoxMode.XcYcWH:
            x, y = boxes[:, 0::4] - boxes[:, 2::4] / 2, boxes[:, 1::4] - boxes[:, 3::4] / 2
            boxes = F.concat((x, y, boxes[:, 2:]), axis=1)
        else:
            raise NotImplementedError

        # XYWH to target type
        if to_mode == BoxMode.XYWH:
            pass
        elif to_mode == BoxMode.XYXY:
            # (x1, y1) should add w and h to get the coordinates of bottom right point
            x1, y1 = boxes[:, 0::4] + boxes[:, 2::4], boxes[:, 1::4] + boxes[:, 3::4]
            boxes = F.concat((boxes[:, :2], x1, y1), axis=1)
        elif to_mode == BoxMode.XcYcWH:
            # (x1, y1) should add w/2 and h/2 to get the coordinates of center point
            xc, yc = boxes[:, 0::4] + boxes[:, 2::4] / 2, boxes[:, 1::4] + boxes[:, 3::4] / 2
            boxes = F.concat((xc, yc, boxes[:, 2:]), axis=1)

        return boxes

    @classmethod
    def numpy_convert(cls, boxes, mode="xywh2xyxy"):
        return np.array(cls.convert(tensor(boxes, device="cpux"), mode).numpy())

    @classmethod
    def get_from_mode_and_to_mode(cls, mode):
        from_mode, to_mode = mode.split("2")
        return cls.parse_mode(from_mode), cls.parse_mode(to_mode)

    @classmethod
    def parse_mode(cls, mode):
        """parse mode string to enum."""
        return {"xyxy": BoxMode.XYXY, "xywh": BoxMode.XYWH, "xcycwh": BoxMode.XcYcWH}[mode.lower()]
