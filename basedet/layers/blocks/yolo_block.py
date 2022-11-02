#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine.functional as F
import megengine.module as M

from basedet.layers import Conv2d

__all__ = ["DepthwiseConvBlock", "SPPBottleneck", "Bottleneck", "CSPLayer"]


class DepthwiseConvBlock(M.Module):
    """Depthwise Conv + Conv"""

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, norm="BN", activation="silu",
    ):
        super().__init__()
        self.dconv = Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, groups=in_channels, bias=False,
            norm=norm, activation=activation,
        )
        self.pconv = Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, groups=1, bias=False,
            norm=norm, activation=activation,
        )

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x


class SPPBottleneck(M.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv2d(
            in_channels, hidden_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )
        self.m = [
            M.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ]
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = Conv2d(
            conv2_channels, out_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.concat([x] + [m(x) for m in self.m], axis=1)
        x = self.conv2(x)
        return x


class Bottleneck(M.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, activation="silu",
    ):
        super().__init__()
        self.use_add = shortcut and in_channels == out_channels

        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv2d(
            in_channels, hidden_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )
        Conv = DepthwiseConvBlock if depthwise else Conv2d
        self.conv2 = Conv(
            hidden_channels, out_channels, 3, stride=1, bias=False,
            padding=1, norm="BN", activation=activation,
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_add:
            y = y + x
        return y


class CSPLayer(M.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels: int, out_channels: int, n: int = 1,
        shortcut: bool = True, expansion: float = 0.5,
        depthwise: bool = False, activation: str = "silu",
    ):
        """
        Args:
            in_channels: input channels.
            out_channels: output channels.
            n: number of Bottlenecks.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = Conv2d(
            in_channels, hidden_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )
        self.conv2 = Conv2d(
            in_channels, hidden_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )
        self.conv3 = Conv2d(
            2 * hidden_channels, out_channels, 1, stride=1, bias=False,
            norm="BN", activation=activation,
        )
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0,
                depthwise, activation=activation,
            ) for _ in range(n)
        ]
        self.m = M.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = F.concat((x_1, x_2), axis=1)
        return self.conv3(x)
