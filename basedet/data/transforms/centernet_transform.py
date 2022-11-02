#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
from typing import Tuple
import cv2
import numpy as np

from megengine.data.transform import VisionTransform

from basedet.utils import registers


@registers.transforms.register()
class CenterAffine(VisionTransform):
    def __init__(self, border=128, output_size=(512, 512), random_aug=True, order=None):
        super().__init__(order)
        self.border = border
        self.output_size = output_size
        self.random_aug = random_aug

    def apply(self, inputs: Tuple):
        assert self.order[0] == "image"
        img_shape = inputs[0].shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        self.affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return super().apply(inputs)

    def _apply_image(self, image):
        return cv2.warpAffine(image, self.affine, self.output_size, flags=cv2.INTER_LINEAR)

    def _apply_coords(self, coords: np.ndarray) -> np.ndarray:
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        # filter logic is in centernet
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    def _apply_mask(self, mask):
        return self._apply_image(mask)

    def generate_center_and_scale(self, img_shape):
        """
        generate center and scale for image randomly

        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, step=0.1))
            h_border = self._get_border(self.border, height)
            w_border = self._get_border(self.border, width)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            raise NotImplementedError("Non-random augmentation not implemented")
        return center, scale

    @classmethod
    def _get_border(cls, border, size):
        """
        decide the border size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= border // i:
            i *= 2
        return border // i

    @classmethod
    def generate_src_and_dst(cls, center, size, output_size):
        """
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])
        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])
        return src, dst


# TODO @wangfeng: Move TestTimeCenterPad to CenterNet Preprocess
@registers.transforms.register()
class TestTimeCenterPad(VisionTransform):

    def __init__(self, order=None):
        super().__init__(order)

    def _apply_image(self, img: np.ndarray) -> np.ndarray:
        h, w, c = img.shape

        target_h, target_w = (h | 31) + 1, (w | 31) + 1
        pad_w, pad_h = math.ceil((target_w - w) / 2), math.ceil((target_h - h) / 2)

        padded_img = np.zeros((target_h, target_w, c))
        padded_img[pad_h:h + pad_h, pad_w:w + pad_w, ...] = img
        return padded_img
