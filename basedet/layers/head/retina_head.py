# -*- coding: utf-8 -*-
import math
from typing import List

import megengine.module as M
from megengine import Tensor


class RetinaNetHead(M.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    """

    def __init__(
        self,
        input_shape,
        num_anchors,
        num_classes,
        strides,
        num_convs=4,
        prior_prob=0.01,
        with_norm=True,
        share_param=True
    ):
        """
        Args:
            input_shape : shape of input.
            num_anchors (int): number of anchors.
            num_classes (int): number of classes.
            strides(List): stride list.
            num_convs (int): number of convs in head.
            prior_prob (float): prior prob of head output.
            with_norm (bool): whether to use normalization.
            share_param (bool): whether to share parameters.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.prior_prob = prior_prob
        self.with_norm = with_norm
        self.share_param = share_param
        self.strides = strides

        cls_subnet = []
        bbox_subnet = []
        in_channels = input_shape[0].channels

        if self.share_param:
            for _ in range(self.num_convs):
                cls_subnet.append(
                    M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                )
                if self.with_norm:
                    cls_subnet.append(M.ReLU())

                bbox_subnet.append(
                    M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                )
                if self.with_norm:
                    bbox_subnet.append(M.ReLU())

            self.cls_subnet = M.Sequential(*cls_subnet)
            self.bbox_subnet = M.Sequential(*bbox_subnet)
            self.cls_score = M.Conv2d(
                in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            )
            self.bbox_pred = M.Conv2d(
                in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
            )
        else:
            self.cls_subnet = [[] for _ in range(len(self.strides))]
            self.bbox_subnet = [[] for _ in range(len(self.strides))]
            self.cls_score = [[] for _ in range(len(self.strides))]
            self.bbox_pred = [[] for _ in range(len(self.strides))]

            for fpn_idx in range(len(input_shape)):
                for _ in range(self.num_convs):
                    self.cls_subnet[fpn_idx].append(
                        M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                    )
                    if with_norm:
                        self.cls_subnet[fpn_idx].append(M.ReLU())

                    self.bbox_subnet[fpn_idx].append(
                        M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                    )
                    if with_norm:
                        self.bbox_subnet[fpn_idx].append(M.ReLU())

                self.cls_subnet[fpn_idx] = M.Sequential(*self.cls_subnet[fpn_idx])
                self.bbox_subnet[fpn_idx] = M.Sequential(*self.bbox_subnet[fpn_idx])

                self.cls_score[fpn_idx] = M.Conv2d(
                    in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
                )
                self.bbox_pred[fpn_idx] = M.Conv2d(
                    in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
                )

        self.init_module()

    def forward(self, features: List[Tensor]):
        logits, offsets = [], []
        for fpn_idx, feature in zip(range(len(self.strides)), features):
            if self.share_param:
                logits.append(self.cls_score(self.cls_subnet(feature)))
                offsets.append(self.bbox_pred(self.bbox_subnet(feature)))
            else:
                logits.append(self.cls_score[fpn_idx](self.cls_subnet[fpn_idx](feature)))
                offsets.append(self.bbox_pred[fpn_idx](self.bbox_subnet[fpn_idx](feature)))
        return logits, offsets

    def init_module(self):
        if self.share_param:
            for modules in [
                self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred
            ]:
                for layer in modules.modules():
                    if isinstance(layer, M.Conv2d):
                        M.init.normal_(layer.weight, mean=0, std=0.01)
                        M.init.fill_(layer.bias, 0)

            # use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            M.init.fill_(self.cls_score.bias, bias_value)
        else:
            for fpn_idx in range(len(self.strides)):
                for modules in [
                    self.cls_subnet[fpn_idx],
                    self.bbox_subnet[fpn_idx],
                    self.cls_score[fpn_idx],
                    self.bbox_pred[fpn_idx]
                ]:
                    for layer in modules.modules():
                        if isinstance(layer, M.Conv2d):
                            M.init.normal_(layer.weight, mean=0, std=0.01)
                            M.init.fill_(layer.bias, 0)

                bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
                M.init.fill_(self.cls_score[fpn_idx].bias, bias_value)
