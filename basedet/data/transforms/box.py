#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Tuple
# import cv2
import numpy as np

from megengine.data.transform.vision import (
    BrightnessTransform,
    ContrastTransform,
    SaturationTransform,
    VisionTransform
)

from basedet.structures.box_utils import get_iou_cpu
from basedet.utils import registers

__all__ = [
    "Expand",
    "MinIoURandomCrop",
    "RandomBrightness",
    "RandomContrast",
    "RandomSaturation",
    "random_wrapper",
]


def random_wrapper(class_name, prob: float = 0.5):
    """
    A class wrapper for transform, return a new transform with random logic.

    Args:
        class_name: Transform type, e.g. `Resize`, `BrightnessTransform`.
        prob: prob of applying transform. Default to 0.5.
    """

    def __init__(self, *args, prob=prob, **kwargs):
        self.prob = prob
        self.__init(*args, **kwargs)

    # code here is really hack, refine them in the future.
    assert not hasattr(class_name, "__init")
    class_name.__init = class_name.__init__
    class_name.__init__ = __init__

    def apply(self, input: Tuple):
        if np.random.random() < self.prob:
            # apply transform
            return self.__apply(input)
        elif not isinstance(input, tuple):
            return (input,)
        else:
            return input

    assert not hasattr(class_name, "__apply")
    class_name.__apply = class_name.apply
    class_name.apply = apply

    return class_name


RandomBrightness = random_wrapper(BrightnessTransform)
registers.transforms.register(RandomBrightness, "RandomBrightness")


RandomContrast = random_wrapper(ContrastTransform)
registers.transforms.register(RandomContrast, "RandomContrast")

RandomSaturation = random_wrapper(SaturationTransform)
registers.transforms.register(RandomSaturation, "RandomSaturation")


@registers.transforms.register()
class MinIoURandomCrop(VisionTransform):
    """
    Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        """
        Args:
            min_ious (tuple): minimum IoU threshold for all intersections with bounding boxes
            min_crop_size (float): minimum crop's size
                (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        """
        super().__init__()
        self.sample_iou = (0, *min_ious, 1)
        self.min_crop_size = min_crop_size

    def get_transform_params(self, input):
        h, w = self._get_image(input).shape[:2]
        boxes = input[self.order.index("boxes")]

        self.do_transform = True
        while True:
            min_iou = np.random.choice(self.sample_iou)
            if min_iou == 1:
                self.do_transform = False
                return

            for i in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))

                overlaps = get_iou_cpu(patch.reshape(-1, 4), boxes)

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1])
                        * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                self.x, self.y, self.w, self.h = int(left), int(top), int(new_w), int(new_h)
                return

    def apply(self, input: Tuple):
        self.get_transform_params(input)
        super().apply(input)

    def _apply_image(self, image):
        return image[..., self.y:self.y + self.h, self.x:self.x + self.w, :]

    def _apply_coords(self, coords):
        coords[:, 0] -= self.x
        coords[:, 1] -= self.y
        return coords

    def _apply_boxes(self, boxes):
        if not self.do_transform:
            return boxes

        boxes = np.array(boxes).reshape(-1, 4)
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (
            (center[:, 0] > self.x) * (center[:, 0] < self.x + self.w)
            * (center[:, 1] > self.y) * (center[:, 1] < self.y + self.h)
        )
        if not mask.any():
            return np.zeros_like(boxes)

        tl = np.array([self.x0, self.y0])
        boxes[:, :2] = np.maximum(boxes[:, :2], tl)
        boxes[:, :2] -= tl

        boxes[:, 2:] = np.minimum(boxes[:, 2:], np.array([self.x0 + self.w, self.y0 + self.h]))
        boxes[:, 2:] -= tl

        return boxes


@registers.transforms.register()
class Expand(VisionTransform):

    def __init__(self, ratio_range=(1, 4), mean=(0, 0, 0), prob=0.5):
        super().__init__()
        self.ratio_range = ratio_range
        self.mean = mean
        self.prob = prob

    def apply(self, input: Tuple):
        self.do_transform = np.random.uniform(0, 1) < self.prob
        return super().apply(input)

    def _apply_image(self, image):
        if self.do_transform:
            h, w, c = image.shape
            ratio = np.random.uniform(*self.ratio_range)
            self.left = int(np.random.uniform(0, w * ratio - w))
            self.top = int(np.random.uniform(0, h * ratio - h))

            expand_img = np.full(
                (int(h * ratio), int(w * ratio), c), self.mean
            ).astype(image.dtype)

            expand_img[self.top:self.top + h, self.left:self.left + w] = image
            expand_img = expand_img.astype("uint8")
            return expand_img
        else:
            image = image.astype("uint8")
            return image

    def _apply_coords(self, coords):
        if self.do_transform:
            coords[:, 0] += self.left
            coords[:, 1] += self.top
        return coords
