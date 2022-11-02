#!/usr/bin/env python3

import megengine as mge
import megengine.functional as F
from megengine import Tensor

from .op_patch import box_center, box_ioa, box_iou


class Boxes(Tensor):
    """
    Boxes is a wrapper for megengine tensor. it provides some basic methods and property of boxes.
    Users could use Boxes like basic megengine tensor.
    """

    # TODO remove mode
    def __init__(self, boxes):
        """
        Args:
            boxes (Tensor): a (N, 4) shape tensor, represented in xyxy mode.
                check :class:`BoxMode` for more information of box mode.
        """
        assert isinstance(boxes, Tensor)
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        self.__dict__ = boxes.__dict__

    @property
    def centers(self):
        """centers of boxes."""
        # NOTE comment next line of code due to mge, uncoment once engine team solve this
        # return (self[:, :2] + self[:, -2:]) / 2.0
        return box_center(self)

    @property
    def area(self):
        """areas of boxes."""
        # w, h = self.width, self.height
        # TODO: assert in the next line will slow down the code, commnet first.
        # assert w.min() >= 0 and h.min() >= 0
        # return w * h
        return self.width * self.height

    @property
    def width(self):
        """width of boxes."""
        return self[:, 2] - self[:, 0]

    @property
    def height(self):
        """width of boxes."""
        return self[:, 3] - self[:, 1]

    def iou(self, boxes):
        """
        Compute the IoU (intersection over union) between all current boxes and given boxes.

        Args:
            boxes (Tensor): boxes tensor with shape (M, 4), box mode must be xyxy also.

        Returns:
            iou (Tensor): IoU matrix, shape (N, M).
        """
        # NOTE comment next 6 line of code due to mge, uncoment once engine team solve this
        # if not isinstance(boxes, Boxes):
        #     boxes = Boxes(boxes)
        # inter = self.intersection(boxes)
        # union = F.expand_dims(self.area, axis=1) + F.expand_dims(boxes.area, axis=0) - inter
        # iou = F.maximum(inter / union, 0)
        # return iou

        return box_iou(self, boxes)

    def giou(self, boxes):
        """
        Compute the GIoU (generalized intersection over union)
            between all current boxes and given boxes.

        Args:
            boxes (Tensor): boxes tensor with shape (M, 4), box mode must be xyxy also.

        Returns:
            giou (Tensor): GIoU matrix, shape (N, M).
        """
        if not isinstance(boxes, Boxes):
            boxes = Boxes(boxes)
        inter = self.intersection(boxes)
        union = F.expand_dims(self.area, axis=1) + F.expand_dims(boxes.area, axis=0) - inter
        iou = inter / union
        box1, box2 = F.expand_dims(self, axis=1), F.expand_dims(boxes, axis=0)
        lt = F.minimum(box1[..., :2], box2[..., :2])
        rb = F.maximum(box1[..., 2:], box2[..., 2:])
        wh = F.clip(rb - lt, lower=0)
        area = wh[..., 0] * wh[..., 1]
        return iou - (area - union) / area

    def ioa(self, boxes) -> Tensor:
        """
        Compute the IoA (intersection over area of given boxes).

        Args:
            boxes (Tensor): boxes tensor with shape (M, 4), box mode must be xyxy also.

        Returns:
            ioa (Tensor): IoU matrix, shape (N, M).
        """
        # NOTE comment next 3 line of code due to mge, uncoment once engine team solve this
        # intersection = self.intersection(boxes)
        # area = F.expand_dims(boxes.area, axis=0)
        # ioa = F.maximum(intersection / area, 0)
        # return ioa
        return box_ioa(self, boxes)

    def intersection(self, boxes: Tensor):
        """
        Compute the intersection between all current boxes and given boxes.

        Args:
            boxes (Tensor): boxes tensor with shape (M, 4), box mode must be xyxy also.

        Returns:
            iou (Tensor): IoU matrix, shape (N, M).
        """
        box1 = F.expand_dims(self, axis=1)
        box2 = F.expand_dims(boxes, axis=0)

        iw = F.minimum(box1[..., 2], box2[..., 2]) - F.maximum(box1[..., 0], box2[..., 0])
        ih = F.minimum(box1[..., 3], box2[..., 3]) - F.maximum(box1[..., 1], box2[..., 1])
        inter = F.maximum(iw, 0) * F.maximum(ih, 0)
        return inter

    def filter_by_size(self, sizes=0):
        """
        Filter boxes by given size. If height/width of box is less than
        given sizes, it will be filtered.

        Args:
            size: (height, width), if a single number is given, it will be casted to 2d tuple.

        Returns:
            kepp_idx(Tensor): indices of keeped boxes.
        """
        if isinstance(sizes, mge.tensor):
            sizes = sizes.tolist()
        if isinstance(sizes, (int, float)):
            sizes = (sizes, sizes)
        assert len(sizes) == 2
        h, w = self.width, self.height
        keep = (w > sizes[0]) & (h > sizes[1])
        return keep

    def clip(self, sizes, inplace=True):
        """Clip the boxes into the image region.

        Args:
            size: image region which value is (height, width)
            inplace (bool): operate boxes inplace or not.
        """
        if isinstance(sizes, mge.tensor):
            sizes = sizes.tolist()
        if isinstance(sizes, (int, float)):
            sizes = (sizes, sizes)
        assert len(sizes) == 2
        h, w = sizes
        # x1 >=0
        box_x1 = F.clip(self[:, 0::4], lower=0, upper=w)
        # y1 >=0
        box_y1 = F.clip(self[:, 1::4], lower=0, upper=h)
        # x2 < w
        box_x2 = F.clip(self[:, 2::4], lower=0, upper=w)
        # y2 < h
        box_y2 = F.clip(self[:, 3::4], lower=0, upper=h)
        clip_boxes = F.concat([box_x1, box_y1, box_x2, box_y2], axis=1)
        if inplace:
            self[:] = clip_boxes
            return self
        return clip_boxes

    def cat(self, boxes, inplace=True):
        """
        Concat boxes.

        Args:
            boxes (Tensor): another boxes to concat.
            inplace (bool): operate in place or not, default: True.
        """
        concat_boxes = F.concat(self, boxes)
        if inplace:
            self[:] = concat_boxes
            return self
        return concat_boxes

    def scale(self, scale_ratios, inplace: bool = True):
        """
        Args:
            scale_ratios: scale ratios of (height, width). If set to 2, the coordinates of boxes
                will multiply 2.
            inplace: operate in place or not, default: True.
        """
        if isinstance(scale_ratios, mge.tensor):
            scale_ratios = scale_ratios.tolist()
        if isinstance(scale_ratios, (int, float)):
            scale_ratios = (scale_ratios, scale_ratios)
        assert len(scale_ratios) == 2
        scale_h, scale_w = scale_ratios

        if inplace:
            self *= mge.tensor([scale_w, scale_h, scale_w, scale_h])
            return self
        else:
            scaled_boxes = self * mge.tensor([scale_w, scale_h, scale_w, scale_h])
            return scaled_boxes

    def __getitem__(self, idx):
        boxes = super().__getitem__(idx)
        # return Boxes type if tensor shape is (N, 4)
        if(boxes.ndim == 2 and boxes.shape[1] == 4):
            boxes = Boxes(boxes)
        return boxes
