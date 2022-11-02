# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import megengine.functional as F
from megengine import Tensor

__all__ = [
    "BoxCoderBase",
    "BoxCoder",
    "PointCoder",
    "SumBoxCoder",
]


class BoxCoderBase(metaclass=ABCMeta):
    """Boxcoder class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def encode(self) -> Tensor:
        pass

    @abstractmethod
    def decode(self) -> Tensor:
        pass


class BoxCoder(BoxCoderBase, metaclass=ABCMeta):
    def __init__(
        self, reg_mean=(0.0, 0.0, 0.0, 0.0), reg_std=(1.0, 1.0, 1.0, 1.0),
    ):
        """
        Args:
            reg_mean(np.ndarray): (x0_mean, x1_mean, y0_mean, y1_mean) or None
            reg_std(np.ndarray):  (x0_std, x1_std, y0_std, y1_std) or None
        """
        self.reg_mean = Tensor(reg_mean, dtype="float32").reshape(1, -1)
        self.reg_std = Tensor(reg_std, dtype="float32").reshape(1, -1)
        super().__init__()

    @staticmethod
    def _box_ltrb_to_cs_opr(bbox, addaxis=None):
        """ transform the left-top right-bottom encoding bounding boxes
        to center and size encodings"""
        bbox_width = bbox[:, 2] - bbox[:, 0]
        bbox_height = bbox[:, 3] - bbox[:, 1]
        bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
        bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
        if addaxis is None:
            return bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y
        else:
            return (
                F.expand_dims(bbox_width, addaxis),
                F.expand_dims(bbox_height, addaxis),
                F.expand_dims(bbox_ctr_x, addaxis),
                F.expand_dims(bbox_ctr_y, addaxis),
            )

    def encode(self, bbox: Tensor, gt: Tensor) -> Tensor:
        bbox_width, bbox_height, bbox_ctr_x, bbox_ctr_y = self._box_ltrb_to_cs_opr(bbox)
        gt_width, gt_height, gt_ctr_x, gt_ctr_y = self._box_ltrb_to_cs_opr(gt)

        target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
        target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
        target_dw = F.log(gt_width / bbox_width)
        target_dh = F.log(gt_height / bbox_height)
        target = F.stack([target_dx, target_dy, target_dw, target_dh], axis=1)

        target -= self.reg_mean
        target /= self.reg_std
        return target

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        deltas *= self.reg_std
        deltas += self.reg_mean

        (
            anchor_width,
            anchor_height,
            anchor_ctr_x,
            anchor_ctr_y,
        ) = self._box_ltrb_to_cs_opr(anchors, 1)
        pred_ctr_x = anchor_ctr_x + deltas[:, 0::4] * anchor_width
        pred_ctr_y = anchor_ctr_y + deltas[:, 1::4] * anchor_height
        pred_width = anchor_width * F.exp(deltas[:, 2::4])
        pred_height = anchor_height * F.exp(deltas[:, 3::4])

        pred_x1 = pred_ctr_x - 0.5 * pred_width
        pred_y1 = pred_ctr_y - 0.5 * pred_height
        pred_x2 = pred_ctr_x + 0.5 * pred_width
        pred_y2 = pred_ctr_y + 0.5 * pred_height

        pred_box = F.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=2)
        pred_box = pred_box.reshape(pred_box.shape[0], -1)

        return pred_box


class SumBoxCoder(BoxCoderBase, metaclass=ABCMeta):

    def __init__(
        self, reg_mean=(0.0, 0.0, 0.0, 0.0), reg_std=(1.0, 1.0, 1.0, 1.0),
    ):
        """
        Args:
            reg_mean(np.ndarray): (x0_mean, x1_mean, y0_mean, y1_mean) or None
            reg_std(np.ndarray):  (x0_std, x1_std, y0_std, y1_std) or None
        """
        self.reg_mean = Tensor(reg_mean, dtype="float32").reshape(1, -1)
        self.reg_std = Tensor(reg_std, dtype="float32").reshape(1, -1)
        super().__init__()

    def encode(self, anchors: Tensor, gt: Tensor) -> Tensor:
        target = gt - anchors

        target -= self.reg_mean
        target /= self.reg_std
        return target

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        deltas *= self.reg_std
        deltas += self.reg_mean

        pred_boxes = anchors + deltas
        return pred_boxes


class PointCoder(BoxCoderBase, metaclass=ABCMeta):

    def encode(self, point: Tensor, gt: Tensor) -> Tensor:
        return F.concat([point - gt[..., :2], gt[..., 2:] - point], axis=-1)

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        return F.stack([
            F.expand_dims(anchors[:, 0], axis=1) - deltas[:, 0::4],
            F.expand_dims(anchors[:, 1], axis=1) - deltas[:, 1::4],
            F.expand_dims(anchors[:, 0], axis=1) + deltas[:, 2::4],
            F.expand_dims(anchors[:, 1], axis=1) + deltas[:, 3::4],
        ], axis=2).reshape(deltas.shape)
