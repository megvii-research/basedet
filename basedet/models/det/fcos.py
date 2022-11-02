#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. All rights reserved.

import numpy as np

import megengine as mge
import megengine.functional as F

from basedet import layers
from basedet.layers import build_backbone, permute_to_N_Any_K
from basedet.models.base_net import BaseNet
from basedet.structures import Boxes, PointCoder
from basedet.utils import all_reduce, registers


@registers.models.register()
class FCOS(BaseNet):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.MODEL

        self.anchor_generator = layers.AnchorPointGenerator(
            model_cfg.ANCHOR.NUM_ANCHORS,
            strides=model_cfg.FPN.STRIDES,
            offset=model_cfg.ANCHOR.OFFSET,
        )
        self.box_coder = PointCoder()

        # build the backbone
        bottom_up = build_backbone(model_cfg.BACKBONE)

        # build FPN
        fpn_cfg = model_cfg.FPN
        self.in_features = fpn_cfg.OUT_FEATURES
        out_channels = fpn_cfg.OUT_CHANNELS

        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=model_cfg.BACKBONE.OUT_FEATURES,
            channels=model_cfg.BACKBONE.OUT_FEATURE_CHANNELS,
            out_channels=out_channels,
            norm=fpn_cfg.NORM,
            top_block=layers.LastLevelP6P7(
                fpn_cfg.TOP_BLOCK_IN_CHANNELS,
                out_channels,
                fpn_cfg.TOP_BLOCK_IN_FEATURE,
            ),
            upsample=fpn_cfg.get("UPSAMPLE", "resize")
        )

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        # build FCOS Head
        head_cfg = model_cfg.HEAD
        num_anchors = [cfg.MODEL.ANCHOR.NUM_ANCHORS] * len(feature_shapes)
        assert len(set(num_anchors)) == 1, "different number of anchors between levels"
        num_anchors = num_anchors[0]
        self.head = layers.PointHead(
            feature_shapes,
            strides=fpn_cfg.STRIDES,
            num_anchors=num_anchors,
            num_classes=cfg.DATA.NUM_CLASSES,
            num_convs=head_cfg.NUM_CONVS,
            prior_prob=head_cfg.CLS_PRIOR_PROB,
            with_norm=head_cfg.get("WITH_NORM", True),
            share_param=head_cfg.get("SHARE_PARAM", True),
        )
        self.num_classes = self.head.num_classes
        img_mean = self.cfg.MODEL.BACKBONE.IMG_MEAN
        if img_mean is not None:
            self.img_mean = mge.tensor(img_mean).reshape(1, -1, 1, 1)
        img_std = self.cfg.MODEL.BACKBONE.IMG_STD
        if img_std is not None:
            self.img_std = mge.tensor(img_std).reshape(1, -1, 1, 1)

    def pre_process(self, inputs):
        image = inputs["data"] if isinstance(inputs, dict) else inputs

        # pad to multiple of 32 and normalize
        image = layers.data_to_input(image, self.img_mean, self.img_std)

        processed_data = {"image": image}
        if self.training:
            processed_data.update(gt_boxes=mge.Tensor(inputs["gt_boxes"]))

        if not isinstance(inputs, dict) or "im_info" not in inputs:
            h, w = image.shape
            img_info = mge.Tensor([[h, w, h, w]])
        else:
            img_info = mge.Tensor(inputs["im_info"])
        processed_data.update(img_info=img_info)

        return processed_data

    def network_forward(self, image: mge.Tensor):
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets, box_ctrness = self.head(features)

        box_logits_list = [permute_to_N_Any_K(x, K=self.num_classes) for x in box_logits]
        box_offsets_list = [permute_to_N_Any_K(x, K=4) for x in box_offsets]
        box_ctrness_list = [permute_to_N_Any_K(x, K=1) for x in box_ctrness]
        anchors_list = self.anchor_generator(features)

        return box_logits_list, box_offsets_list, box_ctrness_list, anchors_list

    def get_losses(self, inputs):
        assert self.training

        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])
        box_logits_list, box_offsets_list, box_ctrness_list, anchors_list = outputs
        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_box_ctrness = F.concat(box_ctrness_list, axis=1)

        img_info = inputs["img_info"]
        gt_boxes = inputs["gt_boxes"]
        gt_labels, gt_offsets, gt_ctrness = self.get_ground_truth(
            anchors_list, gt_boxes, img_info[:, 4].astype("int32"),
        )

        all_level_box_logits = all_level_box_logits.reshape(-1, self.num_classes)
        all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)
        all_level_box_ctrness = all_level_box_ctrness.flatten()

        gt_labels = gt_labels.flatten()
        gt_offsets = gt_offsets.reshape(-1, 4)
        gt_ctrness = gt_ctrness.flatten()

        valid_mask = gt_labels >= 0
        fg_mask = gt_labels > 0
        num_fg = fg_mask.sum()
        sum_ctr = gt_ctrness[fg_mask].sum()
        # add detach() to avoid syncing across ranks in backward
        num_fg = all_reduce(num_fg, mode="mean").detach()
        sum_ctr = all_reduce(sum_ctr, mode="mean").detach()

        gt_targets = F.zeros_like(all_level_box_logits)
        # minus 1 because label 0 represent background in basedet.
        gt_targets[fg_mask, gt_labels[fg_mask] - 1] = 1

        cls_loss = layers.sigmoid_focal_loss(
            all_level_box_logits[valid_mask],
            gt_targets[valid_mask],
            alpha=self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA,
            gamma=self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA,
        ).sum() / F.maximum(1, num_fg)

        bbox_loss = (
            layers.iou_loss(
                all_level_box_offsets[fg_mask],
                gt_offsets[fg_mask],
                box_mode="ltrb",
                loss_type=self.cfg.MODEL.LOSSES.IOU_LOSS_TYPE,
            ) * gt_ctrness[fg_mask]
        ).sum() / F.maximum(1, sum_ctr) * self.cfg.MODEL.LOSSES.REG_LOSS_WEIGHT

        ctr_loss = layers.binary_cross_entropy(
            all_level_box_ctrness[fg_mask],
            gt_ctrness[fg_mask],
            with_logits=True,
        ).sum() / F.maximum(1, num_fg)

        total = cls_loss + bbox_loss + ctr_loss
        loss_dict = {
            "total_loss": total,
            "cls_loss": cls_loss,
            "reg_loss": bbox_loss,
            "ctr_loss": ctr_loss,
        }
        return loss_dict

    def inference(self, inputs):
        assert not self.training
        # currently not support multi-batch testing

        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])
        box_logits_list, box_offsets_list, box_ctrness_list, anchors_list = outputs

        total_scores, logits_label, total_boxes = [], [], []

        for logits, offsets, centerness, anchors in zip(
            box_logits_list, box_offsets_list, box_ctrness_list, anchors_list,
        ):
            scores = F.flatten(F.sqrt(F.sigmoid(logits) * F.sigmoid(centerness)))

            # select all index >= test score in the first, then select topk score value.
            _, keep_idx = layers.non_zeros(scores > self.cfg.TEST.CLS_THRESHOLD)
            if layers.is_empty_tensor(keep_idx):
                continue
            topk_num = min(keep_idx.shape[0], 1000)
            _, topk_idx = F.topk(scores[keep_idx], k=topk_num, descending=True)
            keep_idx = keep_idx[topk_idx]

            total_scores.append(scores[keep_idx])
            logits_label.append(keep_idx % self.num_classes)
            boxes = self.box_coder.decode(anchors, offsets.reshape(-1, 4))
            total_boxes.append(boxes[keep_idx // self.num_classes])

        processed_boxes = self.post_process(
            total_boxes, total_scores, logits_label, inputs["img_info"]
        )
        return processed_boxes

    def post_process(self, *outputs):
        boxes, box_scores, box_labels, img_info = outputs
        return layers.post_process_with_empty_input(
            boxes, box_scores, box_labels, img_info,
            iou_threshold=self.cfg.TEST.IOU_THRESHOLD,
            max_detections_per_image=self.cfg.TEST.MAX_BOXES_PER_IMAGE,
        )

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_valid_gt_box_number):
        labels_list, offsets_list, ctrness_list = [], [], []
        all_level_anchors = F.concat(anchors_list, axis=0)

        center_sampling_radius = self.cfg.MODEL.HEAD.CENTER_SAMPLING_RADIUS
        for boxes_with_labels, num_boxes in zip(batched_gt_boxes, batched_valid_gt_box_number):
            gt_boxes_with_labels = boxes_with_labels[:num_boxes]
            gt_boxes = Boxes(gt_boxes_with_labels[:, :4])

            offsets = self.box_coder.encode(all_level_anchors, F.expand_dims(gt_boxes, axis=1))

            object_sizes_of_interest = F.concat([
                F.broadcast_to(
                    F.expand_dims(mge.tensor(size, dtype=np.float32), axis=0),
                    (anchors_i.shape[0], 2)
                )
                for anchors_i, size in zip(
                    anchors_list,
                    self.cfg.MODEL.HEAD.OBJECT_SIZES_OF_INTEREST,
                )
            ], axis=0)
            max_offsets = F.max(offsets, axis=2)
            is_cared_in_the_level = (
                (max_offsets >= F.expand_dims(object_sizes_of_interest[:, 0], axis=0))
                & (max_offsets <= F.expand_dims(object_sizes_of_interest[:, 1], axis=0))
            )

            if center_sampling_radius > 0:
                gt_centers = gt_boxes.centers
                is_in_boxes = []
                for stride, anchors_i in zip(self.head.strides, anchors_list):
                    radius = stride * center_sampling_radius
                    center_boxes = F.concat([
                        F.maximum(gt_centers - radius, gt_boxes[:, :2]),
                        F.minimum(gt_centers + radius, gt_boxes[:, 2:4]),
                    ], axis=1)
                    center_offsets = self.box_coder.encode(
                        anchors_i, F.expand_dims(center_boxes, axis=1)
                    )
                    is_in_boxes.append(F.min(center_offsets, axis=2) > 0)
                is_in_boxes = F.concat(is_in_boxes, axis=1)
            else:
                is_in_boxes = F.min(offsets, axis=2) > 0

            areas = F.broadcast_to(F.expand_dims(gt_boxes.area, axis=1), offsets.shape[:2])
            areas[~is_cared_in_the_level] = float("inf")
            areas[~is_in_boxes] = float("inf")

            match_indices = F.argmin(areas, axis=0)
            gt_boxes_matched = gt_boxes_with_labels[match_indices]
            anchor_min_area = F.indexing_one_hot(areas, match_indices, axis=0)

            labels = gt_boxes_matched[:, 4].astype("int32")
            labels[anchor_min_area == float("inf")] = 0
            offsets = self.box_coder.encode(all_level_anchors, gt_boxes_matched[:, :4])

            left_right = offsets[:, [0, 2]]
            top_bottom = offsets[:, [1, 3]]
            ctrness = F.sqrt(
                F.maximum(F.min(left_right, axis=1) / F.max(left_right, axis=1), 0)
                * F.maximum(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), 0)
            )

            labels_list.append(labels)
            offsets_list.append(offsets)
            ctrness_list.append(ctrness)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
            F.stack(ctrness_list, axis=0).detach(),
        )
