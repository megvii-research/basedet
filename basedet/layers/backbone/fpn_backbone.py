#!/usr/bin/env python3
from typing import List

import megengine.functional as F
import megengine.module as M

from basedet import layers

__all__ = ["FPN", "LastLevelP6P7", "FPNP6"]


class FPN(M.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps which
    are produced by the backbone networks like ResNet.
    """

    def __init__(
        self,
        bottom_up: M.Module,
        in_features: List[str],
        out_channels: int = 256,
        norm: str = None,
        top_block: M.Module = None,
        strides: tuple = (8, 16, 32),
        channels: tuple = (512, 1024, 2048),
        upsample: str = "resize",
    ):
        """
        Args:
            bottom_up (M.Module): module representing the bottom up sub-network.
                it generates multi-scale feature maps which formatted as a
                dict like {'res3': res3_feature, 'res4': res4_feature}
            in_features (list[str]): list of input feature maps keys coming
                from the `bottom_up` which will be used in FPN.
                e.g. ['res3', 'res4', 'res5']
            out_channels (int): number of channels used in the output
                feature maps.
            norm (str): the normalization type.
            top_block (nn.Module or None): the module build upon FPN layers.
            strides (tuple[int]): strides of input features.
            channels (tuple[int]): input channel size of input features.
        """
        super().__init__()

        in_channels = channels

        use_bias = norm is None
        self.lateral_convs = list()
        self.output_convs = list()
        self.upsample_convs = list()
        stage_index = list(map(lambda x: int(x[-1:]), in_features))
        assert upsample in ["resize", "deconv"]
        self.upsample = upsample

        for idx, in_channels in enumerate(in_channels):
            lateral_norm = layers.get_norm(norm, out_channels)
            output_norm = layers.get_norm(norm, out_channels)

            lateral_conv = layers.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm,
            )
            output_conv = layers.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )

            M.init.msra_normal_(lateral_conv.weight, mode="fan_in")
            M.init.msra_normal_(output_conv.weight, mode="fan_in")

            if use_bias:
                M.init.fill_(lateral_conv.bias, 0)
                M.init.fill_(output_conv.bias, 0)

            stage = int(stage_index[idx])

            setattr(self, "fpn_lateral{}".format(stage), lateral_conv)
            setattr(self, "fpn_output{}".format(stage), output_conv)
            self.lateral_convs.insert(0, lateral_conv)
            self.output_convs.insert(0, output_conv)

            if upsample == "deconv" and idx > 0:
                upsample_conv = M.ConvTranspose2d(
                    out_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
                M.init.msra_normal_(upsample_conv.weight, mode="fan_in")
                setattr(self, "fpn_upsample{}".format(stage), upsample_conv)
                self.upsample_convs.insert(0, upsample_conv)

        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up

        # follow the common practices, FPN features are named to "p<stage>",
        # like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(s): s for s in stage_index
        }

        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(sorted(self._out_feature_strides.keys()))
        self._out_feature_channels = {k: out_channels for k in self._out_features}

    def forward(self, x):
        bottom_up_features = self.bottom_up.extract_features(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]

        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))

        if self.upsample == "deconv":
            for features, lateral_conv, output_conv, upsample_conv in zip(
                x[1:], self.lateral_convs[1:], self.output_convs[1:], self.upsample_convs
            ):
                top_down_features = upsample_conv(prev_features)
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                results.insert(0, output_conv(prev_features))
        else:
            for features, lateral_conv, output_conv in zip(
                x[1:], self.lateral_convs[1:], self.output_convs[1:]
            ):
                top_down_features = F.nn.interpolate(
                    prev_features, scale_factor=2, mode="BILINEAR"
                )
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(
                self.top_block.in_feature, None
            )
            if top_block_in_feature is None:
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))

        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: layers.ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


class FPNP6(M.Module):
    """
    used in FPN, generate a downsampled P6 feature from P5.
    """

    def __init__(self, in_feature="p5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(M.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels: int, out_channels: int, in_feature="c5"):
        super().__init__()
        self.num_levels = 2
        if in_feature == "p5":
            assert in_channels == out_channels
        self.in_feature = in_feature
        self.p6 = M.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = M.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
