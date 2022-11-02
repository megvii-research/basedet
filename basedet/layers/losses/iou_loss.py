#!/usr/bin/python3

import megengine.functional as F
from megengine import Tensor

from basedet.structures import BoxConverter, Boxes, BoxMode


def get_ltrb_boxes_iou(boxes1, boxes2, iou_type="iou", eps: float = 1e-8):
    """
    Get iou value of left-top-right-bottom type boxes.

    Args:
        boxes1 (Tensor): 1st boxes.
        boxes2 (Tensor): 2nd boxes.
        iou_type (str): type of IoU. "iou" and "giou" are supported.
        eps (float): eps value of boxes area.
    """
    assert iou_type in ["iou", "giou"], "iou type {} is not supported".format(iou_type)
    boxes1 = F.concat([-boxes1[..., :2], boxes1[..., 2:]], axis=-1)
    boxes2 = F.concat([-boxes2[..., :2], boxes2[..., 2:]], axis=-1)

    boxes1_area = F.clip(boxes1[..., 2] - boxes1[..., 0], lower=0) * F.clip(
        boxes1[..., 3] - boxes1[..., 1], lower=0
    )
    boxes2_area = F.clip(boxes2[..., 2] - boxes2[..., 0], lower=0) * F.clip(
        boxes2[..., 3] - boxes2[..., 1], lower=0
    )

    w_intersect = F.clip(
        F.minimum(boxes1[..., 2], boxes2[..., 2])
        - F.maximum(boxes1[..., 0], boxes2[..., 0]),
        lower=0,
    )
    h_intersect = F.clip(
        F.minimum(boxes1[..., 3], boxes2[..., 3])
        - F.maximum(boxes1[..., 1], boxes2[..., 1]),
        lower=0,
    )

    area_intersect = w_intersect * h_intersect
    area_union = boxes1_area + boxes2_area - area_intersect
    ious = area_intersect / F.clip(area_union, lower=eps)

    if iou_type == "iou":
        return ious
    elif iou_type == "giou":
        g_w_intersect = F.maximum(boxes1[..., 2], boxes2[..., 2]) - F.minimum(
            boxes1[..., 0], boxes2[..., 0]
        )
        g_h_intersect = F.maximum(boxes1[..., 3], boxes2[..., 3]) - F.minimum(
            boxes1[..., 1], boxes2[..., 1]
        )
        ac_union = g_w_intersect * g_h_intersect
        gious = ious - (ac_union - area_union) / F.clip(ac_union, lower=eps)
        return gious


def iou_loss(
    pred: Tensor,
    target: Tensor,
    box_mode: str = "xyxy",
    loss_type: str = "iou",
    eps: float = 1e-8,
    return_iou: bool = False,
) -> Tensor:
    """
    Get iou loss of predict value and target value.

    Args:
        pred: predicted value.
        target: target value.
        box_mode: mode of boxes. "ltrb", "xyxy", "xywh", "xcycwh" are supported.
        loss_type: type of loss. "iou", "giou", "linear_iou", "square_iou" are supported.
        eps: epsilon value of iou value.
        return_iou: return loss with iou or not.
    """
    assert loss_type in ["iou", "linear_iou", "giou", "square_iou"]

    if box_mode == "ltrb":
        iou_type = "iou" if loss_type == "linear_iou" else loss_type
        ious = get_ltrb_boxes_iou(pred, target, iou_type=iou_type, eps=eps)
    else:
        assert BoxConverter.parse_mode(box_mode) in BoxMode, f"{box_mode} not supported."
        # convert to xyxy mode and compute IoU
        mode = box_mode + "2xyxy"
        pred, target = BoxConverter.convert(pred, mode), BoxConverter.convert(target, mode)
        ious = Boxes(pred).iou(target)

        if loss_type == "giou":
            ious = Boxes(pred).giou(target)
        else:
            ious = Boxes(pred).iou(target)

    if loss_type == "iou":
        loss = -F.log(F.clip(ious, lower=eps))
    elif loss_type == "square_iou":
        loss = 1 - ious ** 2
    else:  # linear_iou and giou
        loss = 1 - ious

    if return_iou:
        return loss, ious

    return loss
