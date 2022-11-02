#!/usr/bin/env python3

import math

import megengine.functional as F
import megengine.module as M

from basedet import layers

__all__ = ["DeconvLayer", "CenternetDeconv", "CenterHead", "SingleHead"]


class DeconvLayer(M.Module):

    def __init__(
        self, in_planes, out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
    ):
        super().__init__()
        if modulate_deform:
            self.dcn = layers.ModulatedDeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn = layers.DeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )

        self.dcn_bn = M.BatchNorm2d(out_planes)
        self.up_sample = M.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            bias=False,
        )
        self.init_module()
        self.up_bn = M.BatchNorm2d(out_planes)
        self.relu = M.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def init_module(self):
        # w = self.up_sample.weight.data
        w = self.up_sample.weight
        f = math.ceil(w.shape[2] / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.shape[0]):
            w[c, 0, :, :] = w[0, 0, :, :]


class CenternetDeconv(M.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, channels, deconv_kernel_sizes, modulate_deform):
        super().__init__()
        # modify into config

        self.num_layers = len(deconv_kernel_sizes)
        for i in range(self.num_layers):
            deconv = DeconvLayer(
                channels[i], channels[i + 1],
                deconv_kernel=deconv_kernel_sizes[i],
                modulate_deform=modulate_deform,
            )
            setattr(self, f"deconv{i+1}", deconv)

    def forward(self, x):
        for i in range(self.num_layers):
            deconv = getattr(self, f"deconv{i+1}")
            x = deconv(x)
        return x


class SingleHead(M.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.feat_conv = M.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = M.ReLU()
        self.out_conv = M.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenterHead(M.Module):

    def __init__(self, in_channels=64, num_classes=80, prior_prob=0.1):
        super().__init__()
        self.cls_head = SingleHead(in_channels, num_classes)
        self.wh_head = SingleHead(in_channels, 2)
        self.reg_head = SingleHead(in_channels, 2)

        self.prior_prob = prior_prob
        self.init_module()

    def init_module(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        M.init.fill_(self.cls_head.out_conv.bias, bias_value)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = F.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            "cls": cls,
            "wh": wh,
            "reg": reg
        }
        return pred
