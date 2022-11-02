#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. All rights reserved.

import math
from typing import List
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.module.normalization import GroupNorm

from basedet import layers


class PointHead(M.Module):
    """
    The head used when anchor points are adopted for object classification and box regression.
    """

    def __init__(
        self,
        input_shape: List[layers.ShapeSpec],
        strides,
        num_anchors,
        num_classes,
        num_convs=4,
        prior_prob=0.01,
        with_norm=True,
        share_param=True,
    ):
        super().__init__()
        self.strides = strides
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.prior_prob = prior_prob
        self.share_param = share_param
        in_channels = input_shape[0].channels

        cls_subnet = []
        bbox_subnet = []
        if share_param:
            for _ in range(self.num_convs):
                cls_subnet.append(
                    M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                )
                if with_norm:
                    cls_subnet.append(GroupNorm(32, in_channels))
                cls_subnet.append(M.ReLU())
                bbox_subnet.append(
                    M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                )
                if with_norm:
                    bbox_subnet.append(GroupNorm(32, in_channels))
                bbox_subnet.append(M.ReLU())

            self.cls_subnet = M.Sequential(*cls_subnet)
            self.bbox_subnet = M.Sequential(*bbox_subnet)
            self.cls_score = M.Conv2d(
                in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            )
            self.bbox_pred = M.Conv2d(
                in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
            )
            self.ctrness = M.Conv2d(
                in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.cls_subnet = [[] for _ in range(len(self.strides))]
            self.bbox_subnet = [[] for _ in range(len(self.strides))]
            self.cls_score = [[] for _ in range(len(self.strides))]
            self.bbox_pred = [[] for _ in range(len(self.strides))]
            self.ctrness = [[] for _ in range(len(self.strides))]
            for fpn_idx in range(len(input_shape)):
                for _ in range(self.num_convs):
                    self.cls_subnet[fpn_idx].append(
                        M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                    )
                    if with_norm:
                        self.cls_subnet[fpn_idx].append(GroupNorm(32, in_channels))
                    self.cls_subnet[fpn_idx].append(M.ReLU())
                    self.bbox_subnet[fpn_idx].append(
                        M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                    )
                    if with_norm:
                        self.bbox_subnet[fpn_idx].append(GroupNorm(32, in_channels))
                    self.bbox_subnet[fpn_idx].append(M.ReLU())
                self.cls_subnet[fpn_idx] = M.Sequential(*self.cls_subnet[fpn_idx])
                self.bbox_subnet[fpn_idx] = M.Sequential(*self.bbox_subnet[fpn_idx])

                self.cls_score[fpn_idx] = M.Conv2d(
                    in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
                )
                self.bbox_pred[fpn_idx] = M.Conv2d(
                    in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
                )
                self.ctrness[fpn_idx] = M.Conv2d(
                    in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1
                )

        self.scales = mge.Parameter(np.ones(len(self.strides), dtype=np.float32))
        self.init_module()

    def init_module(self):
        if self.share_param:
            for modules in [
                self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred, self.ctrness
            ]:
                for layer in modules.modules():
                    if isinstance(layer, M.Conv2d):
                        M.init.normal_(layer.weight, mean=0, std=0.01)
                        M.init.fill_(layer.bias, 0)

            # Use prior in model initialization to improve stability
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            M.init.fill_(self.cls_score.bias, bias_value)
        else:
            for fpn_idx in range(len(self.strides)):
                for modules in [
                    self.cls_subnet[fpn_idx],
                    self.bbox_subnet[fpn_idx],
                    self.cls_score[fpn_idx],
                    self.bbox_pred[fpn_idx],
                    self.ctrness[fpn_idx],
                ]:
                    for layer in modules.modules():
                        if isinstance(layer, M.Conv2d):
                            M.init.normal_(layer.weight, mean=0, std=0.01)
                            M.init.fill_(layer.bias, 0)

                # Use prior in model initialization to improve stability
                bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
                M.init.fill_(self.cls_score[fpn_idx].bias, bias_value)

    def forward(self, features: List[Tensor]):
        logits, offsets, ctrness = [], [], []
        for fpn_idx, (feature, scale, stride) in enumerate(zip(features, self.scales, self.strides)):  # noqa
            if self.share_param:
                logits.append(self.cls_score(self.cls_subnet(feature)))
                bbox_subnet = self.bbox_subnet(feature)
                offsets.append(F.relu(self.bbox_pred(bbox_subnet) * scale) * stride)
                ctrness.append(self.ctrness(bbox_subnet))
            else:
                logits.append(self.cls_score[fpn_idx](self.cls_subnet[fpn_idx](feature)))
                bbox_subnet = self.bbox_subnet[fpn_idx](feature)
                offsets.append(F.relu(self.bbox_pred[fpn_idx](bbox_subnet) * scale) * stride)
                ctrness.append(self.ctrness[fpn_idx](bbox_subnet))

        return logits, offsets, ctrness


class OTAPointHead(PointHead):
    """
    The head used in OTA for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(
        self,
        input_shape: List[layers.ShapeSpec],
        strides,
        num_anchors,
        num_classes,
        norm_reg_targets,
        num_convs=4,
        prior_prob=0.01,
        with_norm=True,
        share_param=True,
    ):
        super().__init__(
            input_shape, strides, num_anchors, num_classes, num_convs,
            prior_prob, with_norm, share_param,
        )
        # rename centerness branch to iou prediction branch
        layers.rename_module(self, "ctrness", "ious_pred")

        self.norm_reg_targets = norm_reg_targets

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #level tensors, each has shape (N, K, H, W).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            offsets (list[Tensor]): #level tensors, each has shape (N, 4, H, W).
                The tensor predicts 4-vector (left, top, right, bottom) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            ious_pred (list[Tensor]): #level tensors, each has shape (N, 1, H, W).
                The tensor predicts the centerness at each spatial position.
        """
        logits, offsets, ious_pred = [], [], []
        # for level, feature in enumerate(features):
        for feature, scale, stride in zip(features, self.scales, self.strides):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)
            logits.append(self.cls_score(cls_subnet))
            ious_pred.append(self.ious_pred(bbox_subnet))

            bbox_pred = scale * (self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                offsets.append(F.relu(bbox_pred) * stride)
            else:
                offsets.append(F.exp(bbox_pred))

        return logits, offsets, ious_pred
