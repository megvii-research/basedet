#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List

import megengine.functional as F
import megengine.module as M

from basedet.layers import Conv2d, CSPLayer, DepthwiseConvBlock, Upsample

__all__ = ["YOLOFPN", "YOLOPAFPN"]


class YOLOFPN(M.Module):

    def __init__(self, bottom_up, in_features, out_channels, bottom_up_channels=None):
        """
        Args:
            bottom_up:
            in_features:
            out_channels:
            bottom_up_channels:
        """
        super().__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels] * len(in_features)

        self.bottom_up = bottom_up
        self.upsample = Upsample()
        self.in_features = in_features

        if bottom_up_channels is None:
            bottom_up_channels = bottom_up.output_shape

        for idx, output_channel in enumerate(out_channels):
            end_idx = None if idx == 0 else -idx
            ch1, ch2 = bottom_up_channels[-(idx + 2):end_idx]

            if idx == 0:
                output_conv = self.build_embedding_layer(ch2, output_channel, [ch1, ch2])
            else:
                output_conv = self.build_embedding_layer(ch1 + ch2, output_channel, [ch1, ch2])
                lateral_conv = self.build_conv_layer(ch2, ch1, ksize=1)
                setattr(self, "lateral_conv{}".format(idx), lateral_conv)
            setattr(self, "output_conv{}".format(idx), output_conv)

    @classmethod
    def build_conv_layer(cls, in_channels, out_channels, ksize):
        return Conv2d(
            in_channels, out_channels, ksize,
            padding=ksize // 2, bias=False,
            norm="BN", activation=M.LeakyReLU(0.1),
        )

    @classmethod
    def build_embedding_layer(cls, in_channels: int, out_channels: int, mid_channels: List[int]):
        embeding = M.Sequential(
            cls.build_conv_layer(in_channels, mid_channels[0], 1),
            cls.build_conv_layer(mid_channels[0], mid_channels[1], 3),
            cls.build_conv_layer(mid_channels[1], mid_channels[0], 1),
            cls.build_conv_layer(mid_channels[0], mid_channels[1], 3),
            cls.build_conv_layer(mid_channels[1], mid_channels[0], 1),
            cls.build_conv_layer(mid_channels[0], mid_channels[1], 3),
            M.Conv2d(mid_channels[1], out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        return embeding

    def forward(self, x):

        def branch_forward(module, x):
            for idx, f in enumerate(module):
                x = f(x)
                if idx == 4:
                    branch_feature = x
            return x, branch_feature

        bottomup_features = self.bottom_up(x)
        features = [bottomup_features[f] for f in self.in_features]

        #  yolo branch 0
        output, branch_output = branch_forward(self.output_conv0, features[-1])

        fpn_out_features = [output]
        for idx in range(1, len(self.in_features)):
            x_in = getattr(self, f"lateral_conv{idx}")(branch_output)
            x_in = self.upsample(x_in)
            x_in = F.concat([x_in, list(reversed(features))[idx]], axis=1)
            output, branch_output = branch_forward(getattr(self, f"output_conv{idx}"), x_in)
            fpn_out_features.append(output)

        return fpn_out_features


class YOLOPAFPN(M.Module):
    """
    PAFPN for YOLO.
    """

    def __init__(
        self, bottom_up, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024], depthwise=False, activation="silu",
    ):
        super().__init__()
        self.backbone = bottom_up
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DepthwiseConvBlock if depthwise else Conv2d

        self.upsample = Upsample(scale_factor=2)
        self.lateral_conv0 = Conv2d(
            int(in_channels[2] * width), int(in_channels[1] * width), 1,
            bias=False, norm="BN", activation=activation,
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            activation=activation,
        )  # cat

        self.reduce_conv1 = Conv2d(
            int(in_channels[1] * width), int(in_channels[0] * width), 1,
            bias=False, norm="BN", activation=activation
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            activation=activation,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3,
            stride=2, padding=1, bias=False,
            norm="BN", activation=activation,
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            activation=activation,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3,
            stride=2, padding=1, bias=False,
            norm="BN", activation=activation,
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            activation=activation,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.

        """

        #  backbone
        out_features = self.backbone(inputs)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = F.concat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = F.concat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = F.concat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = F.concat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
