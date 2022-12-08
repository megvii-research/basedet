#!/usr/bin/env python3

import megengine.module as M

from basedet.layers import Conv2d, Flatten
from basedet.utils import registers


def conv_bn_lrelu(in_channels: int, out_channels: int, ksize: int = 3, stride: int = 1):
    return Conv2d(
        in_channels, out_channels, ksize, stride=stride, bias=False,
        padding=ksize // 2, norm="BN", activation=M.LeakyReLU(0.1),
    )


class DarknetBlock(M.Module):
    """
    Resnet style layer with"in_chs" inputs
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.layer1 = conv_bn_lrelu(in_channels, in_channels // 2, ksize=1)
        self.layer2 = conv_bn_lrelu(in_channels // 2, in_channels, ksize=3)

    def forward(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        return inputs + output


class Darknet(M.Module):
    """
    Darknet megengine implementment
    """
    depth2blocks = {
        21: [1, 1, 2, 2, 1],
        53: [1, 2, 8, 8, 4],
    }

    def __init__(
        self, depth: int, in_channels: int = 3, out_channels: int = 32,
        out_features=None, num_classes=None,
    ):
        """
        Args:
            depth: depth of darknet used in model, support value is in (21, 53).
            in_channels: input channels. Default to 3.
            out_channels: number of filters output in stem
            out_features: desired output layer names, Default to None.
            num_classes: output classs number, for ImageNet, num_classes is 1000. Default to None.
        """
        super().__init__()
        self.stem = conv_bn_lrelu(in_channels, out_channels, ksize=3, stride=1)
        self.num_classes = num_classes

        current_stride = 1
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": out_channels}

        self.stages_and_names = []
        num_blocks = Darknet.depth2blocks[depth]
        self.output_shape = []

        for i, block_value in enumerate(num_blocks):
            name = "dark" + str(i + 1)
            stage = self.make_stage(out_channels, block_value, stride=2)
            setattr(self, name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride
            current_stride *= 2
            out_channels *= 2
            self._out_feature_channels[name] = out_channels
            self.output_shape.append(out_channels)

        if num_classes is not None:
            name = "linear"
            self.linear = M.Sequential([
                M.AdaptiveAvgPool2d(1),
                Flatten(),
                M.Linear(in_features=out_channels, out_features=num_classes)
            ])

        if out_features is None:
            out_features = [name]
        self.out_features = out_features

    def forward(self, inputs):
        outputs = {}
        inputs = self.stem(inputs)

        if "stem" in self.out_features:
            outputs["stem"] = inputs
        for stage, name in self.stages_and_names:
            inputs = stage(inputs)
            if name in self.out_features:
                outputs[name] = inputs
        if self.num_classes is not None:
            inputs = self.linear(inputs)
            if "linear" in self.out_features:
                outputs["linear"] = inputs
        return outputs

    @classmethod
    def make_stage(cls, in_channels, num_blocks, stride=1):
        group_layer = [
            conv_bn_lrelu(in_channels, in_channels * 2, stride=stride)
        ] + [(DarknetBlock(in_channels * 2)) for i in range(num_blocks)]
        return M.Sequential(*group_layer)


@registers.models.register()
def darknet21(**kwargs):
    return Darknet(depth=21, **kwargs)


@registers.models.register()
def darknet53(**kwargs):
    return Darknet(depth=53, **kwargs)
