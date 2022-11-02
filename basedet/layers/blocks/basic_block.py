#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# this file includes functions of module level operator and module wrapper

import megengine.functional as F
import megengine.module as M

from basedet.layers import Conv2d

__all__ = ["Focus", "Flatten", "Upsample"]


class Focus(M.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, activation="silu"):
        super().__init__()
        self.conv = Conv2d(
            in_channels * 4, out_channels, ksize, stride=stride, bias=False,
            padding=ksize // 2, norm="BN", activation=activation,
        )

    def forward(self, x):
        # shape changed from (b, c, w, h) to (b, 4c, w/2, h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = F.concat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), axis=1,
        )
        return self.conv(x)


class Flatten(M.Module):
    """
    Faltten input to a single dimension
    """

    def __init__(self, start_axis=0, end_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

    def forward(self, x):
        x = F.flatten(x, self.start_axis, self.end_axis)
        return x

    def _module_info_string(self) -> str:
        return "start_axis={start_axis}, end_axis={end_axis}".format(**self.__dict__)


class Upsample(M.Module):

    def __init__(self, scale_factor: float = 2.0, mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.vision.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

    def _module_info_string(self) -> str:
        return "scale_factor={scale_factor}, mode={mode}".format(**self.__dict__)
