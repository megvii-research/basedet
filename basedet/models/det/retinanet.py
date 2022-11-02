#!/usr/bin/python3
# Copyright (c) Megvii, Inc. All rights reserved.

import megengine as mge
import megengine.functional as F

from basedet import layers
from basedet.layers import data_to_input, permute_to_N_Any_K
from basedet.models.base_net import BaseNet
from basedet.structures import BoxCoder, Boxes
from basedet.utils import registers


@registers.models.register()
class RetinaNet(BaseNet):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg.MODEL
        anchor_scales = model_cfg.ANCHOR.SCALES
        anchor_ratios = model_cfg.ANCHOR.RATIOS
        self.anchor_gen = layers.DefaultAnchorGenerator(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            strides=model_cfg.FPN.STRIDES,
            offset=model_cfg.ANCHOR.OFFSET,
        )
        self.box_coder = BoxCoder(model_cfg.BOX_REG.MEAN, model_cfg.BOX_REG.STD)
        self.matcher = layers.Matcher(
            model_cfg.MATCHER.THRESHOLDS,
            model_cfg.MATCHER.LABELS,
            model_cfg.MATCHER.ALLOW_LOW_QUALITY
        )
        self.in_features = model_cfg.FPN.OUT_FEATURES

        # build backbone
        bottom_up = layers.build_backbone(model_cfg.BACKBONE)

        # build RetinaNet FPN
        fpn_cfg = model_cfg.FPN
        out_channels = model_cfg.FPN.OUT_CHANNELS
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

        # build RetinaNet Head
        if len(anchor_ratios) != len(anchor_scales):
            anchor_ratios = anchor_ratios * len(anchor_scales)
        num_anchors = [
            len(anchor_scales[i]) * len(anchor_ratios[i]) for i in range(len(feature_shapes))
        ]
        assert (len(set(num_anchors)) == 1), "different number of anchors between levels"
        num_anchors = num_anchors[0]
        head_cfg = model_cfg.HEAD
        self.head = layers.RetinaNetHead(
            feature_shapes,
            num_anchors,
            num_classes=cfg.DATA.NUM_CLASSES,
            strides=fpn_cfg.STRIDES,
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
        image = data_to_input(image, self.img_mean, self.img_std)

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
        box_logits, box_offsets = self.head(features)

        box_logits_list = [permute_to_N_Any_K(x, K=self.num_classes) for x in box_logits]
        box_offsets_list = [permute_to_N_Any_K(x, K=4) for x in box_offsets]
        anchors_list = self.anchor_gen(features)

        return box_logits_list, box_offsets_list, anchors_list

    def get_losses(self, inputs):
        assert self.training

        # get output meta data
        inputs = self.pre_process(inputs)
        box_logits_list, box_offsets_list, anchors_list = self.network_forward(inputs["image"])

        all_level_box_logits = F.concat(box_logits_list, axis=1)
        all_level_box_offsets = F.concat(box_offsets_list, axis=1)
        all_level_anchors = F.concat(anchors_list, axis=0)

        all_level_box_logits = all_level_box_logits.reshape(-1, self.num_classes)
        all_level_box_offsets = all_level_box_offsets.reshape(-1, 4)
        img_info = inputs["img_info"]
        gt_boxes = inputs["gt_boxes"]

        # get decoded ground truth
        gt_labels, gt_offsets = self.get_ground_truth(
            all_level_anchors, gt_boxes, img_info[:, 4].astype("int32"),
        )

        gt_labels = gt_labels.flatten().astype("int32")
        gt_offsets = gt_offsets.reshape(-1, 4)

        valid_mask = gt_labels >= 0
        fg_mask = gt_labels > 0
        num_fg = fg_mask.sum()

        gt_targets = F.zeros_like(all_level_box_logits)
        gt_targets[fg_mask, (gt_labels[fg_mask] - 1)] = 1

        cls_loss = layers.sigmoid_focal_loss(
            all_level_box_logits[valid_mask],
            gt_targets[valid_mask],
            alpha=self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA,
            gamma=self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA,
        ).sum() / F.maximum(1, num_fg)

        bbox_loss = layers.smooth_l1_loss(
            all_level_box_offsets[fg_mask],
            gt_offsets[fg_mask],
            beta=self.cfg.MODEL.LOSSES.SMOOTH_L1_BETA,
        ).sum() / F.maximum(1, num_fg) * self.cfg.MODEL.LOSSES.REG_LOSS_WEIGHT

        total_loss = cls_loss + bbox_loss
        loss_dict = {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "reg_loss": bbox_loss,
        }
        return loss_dict

    def inference(self, inputs):
        assert not self.training
        # currently not support multi-batch testing during inference
        inputs = self.pre_process(inputs)
        logits_list, offsets_list, anchors_list = self.network_forward(inputs["image"])

        total_scores = []
        logits_label = []
        total_boxes = []
        for logits, offsets, anchors in zip(logits_list, offsets_list, anchors_list):
            num_boxes, num_classes = logits.shape[-2:]
            scores = F.sigmoid(F.flatten(logits))

            # select all index >= test score in the first, then select topk score value.
            _, keep_idx = layers.non_zeros(scores > self.cfg.TEST.CLS_THRESHOLD)
            if layers.is_empty_tensor(keep_idx):
                continue
            topk_num = min(keep_idx.shape[0], 1000)
            _, topk_idx = F.topk(scores[keep_idx], k=topk_num, descending=True)
            keep_idx = keep_idx[topk_idx]

            total_scores.append(scores[keep_idx])
            logits_label.append(keep_idx % num_classes)
            boxes = self.box_coder.decode(anchors, offsets.reshape(-1, 4))
            total_boxes.append(boxes[keep_idx // num_classes])

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

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_valid_gt_box_number):
        labels_list = []
        offsets_list = []

        for boxes_with_labels, num_boxes in zip(batched_gt_boxes, batched_valid_gt_box_number):
            gt_boxes = boxes_with_labels[:num_boxes]

            overlaps = Boxes(gt_boxes[:, :4]).iou(Boxes(anchors))
            match_indices, labels = self.matcher(overlaps)
            gt_boxes_matched = gt_boxes[match_indices]

            fg_mask = labels == 1
            labels[fg_mask] = gt_boxes_matched[fg_mask, 4].astype("int32")
            offsets = self.box_coder.encode(anchors, gt_boxes_matched[:, :4])

            labels_list.append(labels)
            offsets_list.append(offsets)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
        )
