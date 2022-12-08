#!/usr/bin/env python3

import megengine.module as M

from basedet.layers import Conv2d, CSPLayer, DepthwiseConvBlock, Focus, SPPBottleneck
from basedet.utils import registers


# TODO @wangfeng: unite 2 types of DarkNet
class CSPDarknet(M.Module):

    def __init__(
        self, depth_factor, width_factor,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False, activation="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DepthwiseConvBlock if depthwise else Conv2d

        base_depth = max(round(depth_factor * 3), 1)  # 3
        base_channels = int(width_factor * 64)  # 64

        # stem
        self.stem = Focus(3, base_channels, ksize=3, activation=activation)

        # dark2
        self.dark2 = M.Sequential(
            Conv(
                base_channels, base_channels * 2, 3, stride=2, bias=False,
                padding=1, norm="BN", activation=activation,
            ),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, activation=activation
            ),
        )

        # dark3
        self.dark3 = M.Sequential(
            Conv(
                base_channels * 2, base_channels * 4, 3, stride=2, bias=False,
                padding=1, norm="BN", activation=activation
            ),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, activation=activation,
            ),
        )

        # dark4
        self.dark4 = M.Sequential(
            Conv(
                base_channels * 4, base_channels * 8, 3, stride=2, bias=False,
                padding=1, norm="BN", activation=activation,
            ),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, activation=activation,
            ),
        )

        # dark5
        self.dark5 = M.Sequential(
            Conv(
                base_channels * 8, base_channels * 16, 3, stride=2, bias=False,
                padding=1, norm="BN", activation=activation,
            ),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=activation),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, activation=activation,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


@registers.models.register()
def csp_darknet(*args, **kwargs):
    return CSPDarknet(*args, **kwargs)
