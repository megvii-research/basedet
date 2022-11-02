#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
from typing import List

import megengine as mge
import megengine.module as M

from basedet import layers

__all__ = ["YOLOHead", "YOLOXHead"]


class YOLOHead(M.Module):

    def __init__(self, output_dim, num_anchors):
        super().__init__()
        self.output_dim = output_dim
        self.num_anchors = num_anchors

    def forward(self, features: List[mge.Tensor]):
        pred = []

        for x, num_anchor in zip(features, self.num_anchors):
            batchsize, _, h, w = x.shape

            prediction = x.reshape(
                batchsize, num_anchor, self.output_dim, h, w
            ).transpose(0, 1, 3, 4, 2)
            pred.append(prediction)

        return pred


class YOLOXHead(M.Module):

    """Decoupled head in YOLOX"""

    def __init__(
        self,
        num_classes,
        in_channels=[256, 512, 1024],
        mid_channels: int = 256,
        act: str = "silu",
        depthwise: bool = False,
        prior_prob: float = 0.01,
    ):
        """
        Args:
            act: activation type of conv. Defalut to "silu".
            depthwise: wheather apply depthwise convblock in conv branch. Defalut to False.
        """
        super().__init__()
        self.num_anchors = 1
        self.num_classes = num_classes
        self.prior_prob = prior_prob

        # network structure
        self.stems = []
        self.cls_convs, self.cls_preds = [], []
        self.reg_convs, self.reg_preds = [], []
        self.obj_preds = []
        Conv = layers.DepthwiseConvBlock if depthwise else layers.Conv2d

        for channels in in_channels:
            self.stems.append(
                layers.Conv2d(
                    channels, mid_channels, kernel_size=1,
                    bias=False, norm="BN", activation=act,
                ),
            )
            self.cls_convs.append(
                M.Sequential(*[
                    Conv(
                        mid_channels, mid_channels, kernel_size=3, padding=1,
                        bias=False, norm="BN", activation=act,
                    ) for _ in range(2)
                ])
            )
            self.reg_convs.append(
                M.Sequential(*[
                    Conv(
                        mid_channels, mid_channels, kernel_size=3,
                        bias=False, padding=1, norm="BN", activation=act,
                    ) for _ in range(2)
                ])
            )

            self.cls_preds.append(
                M.Conv2d(mid_channels, self.num_anchors * self.num_classes, kernel_size=1)
            )
            self.reg_preds.append(M.Conv2d(mid_channels, 4, kernel_size=1))
            self.obj_preds.append(M.Conv2d(mid_channels, self.num_anchors, kernel_size=1))

        self.init_module()

    def init_module(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.cls_preds, self.obj_preds]:
            for conv in module:
                M.init.fill_(conv.bias, bias_value)

    def forward(self, featuers):
        logits, offsets, objs = [], [], []

        for level, x in enumerate(featuers):
            x = self.stems[level](x)

            pred_logits = self.cls_convs[level](x)
            pred_logits = self.cls_preds[level](pred_logits)

            reg_feature = self.reg_convs[level](x)
            pred_offset = self.reg_preds[level](reg_feature)
            pred_obj = self.obj_preds[level](reg_feature)
            del reg_feature

            logits.append(pred_logits)
            offsets.append(pred_offset)
            objs.append(pred_obj)

        return logits, offsets, objs
