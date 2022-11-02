#!/usr/bin/env python3

from typing import Optional
import numpy as np

import megengine.functional as F
from megengine import Tensor

__all__ = [
    "batched_nms",
    "post_processing",
    "py_cpu_nms",
    "post_process_with_empty_input",
]


def batched_nms(
    boxes: Tensor, scores: Tensor, idxs: Tensor, iou_thresh: float, max_output: Optional[int] = None
) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according to
    their intersection-over-union (IoU).

    Args:
        boxes: shape `(N, 4)`, the boxes to perform nms on each box
            is expected to be in `(x1, y1, x2, y2)` format.
        iou_thresh: ``IoU`` threshold for overlapping.
        idxs: shape `(N,)`, the class indexs of boxes in the batch.
        scores: shape `(N,)`, the score of boxes.

    Returns:
        indices of the elements that have been kept by NMS.
    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert scores.ndim == 1, "the expected shape of scores is (N,)"
    assert idxs.ndim == 1, "the expected shape of idxs is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0] == idxs.shape[0]
    ), "number of boxes, scores and idxs are not matched"

    idxs = idxs.detach()
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes = boxes + offsets.reshape(-1, 1)
    return F.vision.nms(boxes, scores, iou_thresh, max_output)


def post_process_with_empty_input(
    boxes: Tensor,
    box_scores: Tensor,
    box_labels: Tensor,
    img_info: Tensor,
    iou_threshold: float = 0.5,
    max_detections_per_image: int = 100,
):
    from basedet.structures import Boxes, Container
    if not boxes:
        empty_tensor = Tensor([])
        return Container(boxes=empty_tensor, box_scores=empty_tensor, box_labels=empty_tensor)

    boxes_container = Container(
        boxes=Boxes(F.concat(boxes, axis=0)),
        box_scores=F.concat(box_scores, axis=0),
        box_labels=F.concat(box_labels, axis=0),
    )

    processed_boxes = post_processing(
        boxes_container, img_info,
        iou_threshold=iou_threshold,
        max_detections_per_image=max_detections_per_image,
    )
    return processed_boxes


# TODO support general post_process logic, currently specific logic for COCO
def post_processing(
    boxes_container,
    img_info,
    iou_threshold,
    process_method="nms",
    max_detections_per_image=None,
):
    """
    post processing
    """
    # nms operation guarantees socres of boxes sorted in desecend order.
    keep_idx = batched_nms(
        boxes_container.boxes,
        boxes_container.box_scores,
        boxes_container.box_labels,
        iou_thresh=iou_threshold,
        max_output=max_detections_per_image
    )
    keeped_boxes = boxes_container[keep_idx]

    # scale image to image size
    scale_ratios = img_info[0, 2] / img_info[0, 0], img_info[0, 3] / img_info[0, 1]
    # clipped_boxes = Boxes(boxes)
    keeped_boxes.boxes.scale(scale_ratios).clip(img_info[0, 2:4])

    return keeped_boxes


def py_cpu_nms(dets: np.ndarray, thresh: float):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        # add eps = 1e-5 here to avoid nan value
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep
