#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine.functional as F
import megengine.module as M

__all__ = ["DeformConvWithOff", "ModulatedDeformConvWithOff"]


class DeformConvWithOff(M.Module):

    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
        dilation=1, deformable_groups=1
    ):
        super().__init__()
        self.offset_conv = M.Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcn = M.DeformableConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=deformable_groups)

    def forward(self, x):
        offset = self.offset_conv(x)
        output = self.dcn(
            x, offset, mask=F.ones_like(offset)[:, :offset.shape[1] // 2, ...],
        )
        return output


class ModulatedDeformConvWithOff(M.Module):

    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1,
        dilation=1, deformable_groups=1
    ):
        super().__init__()
        self.offset_mask_conv = M.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = M.DeformableConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=deformable_groups
        )

    def forward(self, inputs):
        x = self.offset_mask_conv(inputs)
        o1, o2, mask = F.split(x, 3, axis=1)
        offset = F.concat((o1, o2), axis=1)
        mask = F.sigmoid(mask)
        output = self.dcnv2(inputs, offset, mask)
        return output
