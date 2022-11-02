#!/usr/bin/python3
# -*- coding:utf-8 -*-
from scipy.optimize import linear_sum_assignment

import megengine as mge
import megengine.functional as F
import megengine.module as M

from ..blocks import SinkhornDistance
from ..losses import iou_loss

__all__ = [
    "Matcher",
    "HungarianMatcher",
    "SinkhornMatcher",
    "OTATopkMatcher",
]


class Matcher:

    def __init__(self, thresholds, labels, allow_low_quality_matches=False):
        assert len(thresholds) + 1 == len(labels), "thresholds and labels are not matched"
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        thresholds.append(float("inf"))
        thresholds.insert(0, -float("inf"))

        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, matrix):
        """
        matrix(tensor): A two dim tensor with shape of (N, M). N is number of GT-boxes,
            while M is the number of anchors in detection.
        """
        assert len(matrix.shape) == 2
        max_scores = matrix.max(axis=0)
        match_indices = F.argmax(matrix, axis=0)

        # default ignore label: -1
        labels = F.full_like(match_indices, -1)

        for label, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            mask = (max_scores >= low) & (max_scores < high)
            labels[mask] = label

        if self.allow_low_quality_matches:
            mask = (matrix == F.max(matrix, axis=1, keepdims=True)).sum(axis=0) > 0
            labels[mask] = 1

        return match_indices, labels


class HungarianMatcher(M.Module):

    def __init__(self, weight_class=1.0, weight_bbox=1.0, weight_giou=1.0):
        super().__init__()
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        assert (
            weight_class != 0 or weight_bbox != 0 or weight_giou != 0
        ), "all 0 weights"

    def forward(self, outputs, targets):
        B, N = outputs["pred_logits"].shape[:2]

        out_prob = F.softmax(F.flatten(outputs["pred_logits"], 0, 1))
        out_bbox = F.flatten(outputs["pred_boxes"], 0, 1)

        tgt_ids = F.concat([v["boxes_category"] for v in targets])
        tgt_bbox = F.concat([v["boxes"] for v in targets])
        if len(tgt_bbox) == 0:
            return [
                (mge.tensor((), dtype="int32"), mge.tensor((), dtype="int32"))
                for _ in targets
            ]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = F.abs(F.expand_dims(out_bbox, 1) - F.expand_dims(tgt_bbox, 0)).sum(
            -1
        )
        cost_giou = iou_loss(out_bbox, tgt_bbox, "xcycwh", "giou") - 1

        C = (
            self.weight_bbox * cost_bbox
            + self.weight_class * cost_class
            + self.weight_giou * cost_giou
        ).reshape(B, N, -1)

        indices, st = [], 0
        for i, v in enumerate(targets):
            ed = st + v["boxes"].shape[0]
            if st == ed:
                indices.append(((), ()))
            else:
                indices.append(linear_sum_assignment(C[i, :, st:ed]))
            st = ed

        return [
            (mge.tensor(_I, dtype="int32"), mge.tensor(J, dtype="int32"))
            for _I, J in indices
        ]


class SinkhornMatcher:

    def __init__(self, eps=0.1, max_iter=50):
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=50)

    def __call__(self, cost: mge.Tensor, ious: mge.Tensor):
        _, num_anchors = cost.shape
        topk_ious, _ = F.topk(ious, k=20, descending=True)
        mu = F.clip(topk_ious.sum(axis=1).astype("int32"), lower=1).astype("float32")
        mu = F.concat((mu, num_anchors - mu.sum()), axis=0)
        nu = F.ones(num_anchors)

        # optimal transportation plan pi
        _, pi = self.sinkhorn(mu, nu, cost)

        # rescale pi so that the max pi for each gt equals to 1.
        rescale_factor = pi.max(axis=1)
        pi = pi / F.expand_dims(rescale_factor, axis=1)

        matched_gt_inds = F.argmax(pi, axis=0)
        return matched_gt_inds


class OTATopkMatcher:

    def __init__(self, candidate_k=10):
        self.candidate_k = candidate_k

    def __call__(self, cost: mge.Tensor, ious: mge.Tensor):
        """
        Args:
            cost: cost matrix, shape (#boxes, #anchors).
            ious: pairwise iou of gt boxes and anchors, shape (#boxes, #anchors).
        """
        num_boxes, num_anchors = cost.shape
        matching_matrix = F.zeros((num_boxes, num_anchors))

        topk_ious, _ = F.topk(ious, self.candidate_k, descending=True)
        dynamic_ks = F.clip(topk_ious.sum(1).astype("int32"), lower=1)
        for gt_idx in range(num_boxes):
            _, anchor_idx = F.topk(cost[gt_idx], k=dynamic_ks[gt_idx], descending=False)
            matching_matrix[gt_idx, anchor_idx] = 1.0

        del topk_ious, dynamic_ks, anchor_idx

        # one single anchor should only match one gt boxes
        multi_match = matching_matrix.sum(0) > 1
        if (multi_match).sum() > 0:
            cost_argmin = F.argmin(cost[:, multi_match], axis=0)
            matching_matrix[:, multi_match] = 0.0
            matching_matrix[cost_argmin, multi_match] = 1.0

        default_match = F.ones((1, num_anchors))
        matching_matrix = F.concat((matching_matrix * 2, default_match), axis=0)

        return F.argmax(matching_matrix, axis=0)
