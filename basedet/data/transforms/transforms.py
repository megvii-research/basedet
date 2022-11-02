#!/usr/bin/env python3
from typing import Tuple
import cv2
import numpy as np

from megengine.data.transform import Compose, Resize, ShortestEdgeResize, VisionTransform

from basedet.utils.registry import registers

__all__ = [
    "PadToTargetSize",
    "ToMode",
    "TestTimeCompose",
    "RandomSizeCrop",
]


@registers.transforms.register()
class PadToTargetSize(VisionTransform):
    """Pad the input image to target size. Only pad on the right and bottom.

    Args:
        target_size: padding size of input image, it could be integer or sequence.
            If it is an integer, the input image will be padded in square shape.
            If it is a sequence containing two integers, it should be value of (height, width).
        pad_value(int, optional): padding value of image, could be a sequence of int or float.
            if it is float value, the dtype of image will be casted to float32.  Defaults to 0.
        order: the same with :class:`VisionTransform`.
    """

    def __init__(self, target_size, pad_value=0, mask_value=0, *, order=None):
        super().__init__(order)
        self.target_size = target_size
        self.pad_value = pad_value
        self.mask_value = mask_value

    def _apply_image(self, image: np.array):
        h, w, _ = image.shape
        pad_size = (0, self.target_size[0] - h, 0, self.target_size[1] - w)
        self.pad_size = pad_size
        return cv2.copyMakeBorder(
            image, *self.pad_size, cv2.BORDER_CONSTANT, value=self.pad_value,
        )

    def _apply_coords(self, coords):
        # only pad on left and bottom, so it has no side effect of coordinates.
        return coords

    def _apply_mask(self, mask):
        return cv2.copyMakeBorder(
            mask, *self.pad_size, cv2.BORDER_CONSTANT, value=self.mask_value,
        )


@registers.transforms.register()
class ToMode(VisionTransform):
    r"""Change input data to a target mode.
    For example, most transforms use HWC mode image,
    while the neural network might use CHW mode input tensor.

    Args:
        mode: output mode of input, support. Default to "CHW".
        order: the same with :class:`VisionTransform`
    """

    def __init__(self, mode="CHW", *, order=None):
        super().__init__(order)
        assert mode in ["CHW", "NCHW"], "unsupported mode: {}".format(mode)
        self.mode = mode

    def __call__(self, *args, **kwargs):
        return self.apply(args, kwargs)

    def _apply_image(self, image):
        # transpose is faster than rollaxis
        if self.mode == "CHW":
            return np.ascontiguousarray(image.transpose(2, 0, 1))
        elif self.mode == "NCHW":
            return np.ascontiguousarray(image.transpose(2, 0, 1)[None, :, :, :], dtype=np.float32)

        return image

    def _apply_coords(self, coords):
        return coords

    def _apply_mask(self, mask):
        return self._apply_image(mask)


@registers.transforms.register()
class TestTimeCompose(Compose):
    """Compose transforms only for test time. This class will generate img_info automatically."""

    def __iter__(self):
        for t in self.transforms:
            yield t

    def __call__(self, image) -> Tuple[np.ndarray, np.array]:
        """NOTE: TestTimeCompose only support image input."""
        original_h, original_w, _ = image.shape
        aug_shape = None

        for aug in iter(self):
            image = aug.apply(image)
            if self.is_shape_from(aug):
                aug_shape = aug._shape_info[2:]

        if aug_shape is None:
            # get the shape from converted image
            aug_shape = image.shape[-2:]
        im_info = np.array([(*aug_shape, original_h, original_w)], dtype=np.float32,)

        return image, im_info

    def is_shape_from(self, aug) -> bool:
        # TODO wangfeng: refine this function
        return isinstance(
            aug,
            (
                Resize, ShortestEdgeResize,
            )
        )


@registers.transforms.register()
class RandomSizeCrop(VisionTransform):
    def __init__(self, min_size, max_size, *, order=None):
        super().__init__(order)
        self.min_size = min_size
        self.max_size = max_size

    def apply(self, input: Tuple):
        self._h, self._w, _ = self._get_image(input).shape
        self._th = np.random.randint(self.min_size, min(self._h, self.max_size))
        self._tw = np.random.randint(self.min_size, min(self._w, self.max_size))
        self._x = np.random.randint(0, max(0, self._w - self._tw) + 1)
        self._y = np.random.randint(0, max(0, self._h - self._th) + 1)
        return super().apply(input)

    def _apply_image(self, image):
        return image[self._y: self._y + self._th, self._x: self._x + self._tw]

    def _apply_coords(self, coords):
        coords[:, 0] -= self._x
        coords[:, 1] -= self._y
        return coords

    def _apply_boxes(self, boxes):
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(boxes).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self._apply_coords(coords).reshape((-1, 4, 2))
        coords[:, :, 0] = coords[:, :, 0].clip(min=0., max=self._tw)
        coords[:, :, 1] = coords[:, :, 1].clip(min=0., max=self._th)
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        cropped_boxes = trans_boxes.reshape(-1, 2, 2)
        self._keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        cropped_boxes = trans_boxes[self._keep]
        return cropped_boxes

    def _apply_boxes_category(self, boxes_category):
        return boxes_category[self._keep]
