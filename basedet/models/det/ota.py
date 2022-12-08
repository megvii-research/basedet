#!/usr/bin/env python3

import numpy as np

import megengine.functional as F

from basedet import layers
from basedet.configs import OTAConfig
from basedet.layers import permute_to_N_Any_K
from basedet.utils import all_reduce, registers

from .fcos import FCOS


@registers.models.register()
class OTA(FCOS):

    def __init__(self, cfg: OTAConfig):
        super().__init__(cfg)
        self.matching = cfg.MODEL.MATCHING
        assert self.matching in ("topk", "sinkhorn"), f"unsupported matching named {self.matching}"
        self.reg_weight = cfg.MODEL.HEAD.get("COST_REG_WEIGHTS", 1.5)

        # rebuild the OTA Head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        head_cfg = cfg.MODEL.HEAD
        num_anchors = [cfg.MODEL.ANCHOR.NUM_ANCHORS] * len(feature_shapes)
        assert len(set(num_anchors)) == 1, "different number of anchors between levels"
        num_anchors = num_anchors[0]
        self.head = layers.OTAPointHead(
            feature_shapes,
            strides=cfg.MODEL.FPN.STRIDES,
            num_anchors=num_anchors,
            num_classes=cfg.DATA.NUM_CLASSES,
            num_convs=head_cfg.NUM_CONVS,
            prior_prob=head_cfg.CLS_PRIOR_PROB,
            norm_reg_targets=head_cfg.NORM_REG_TARGETS,
            with_norm=head_cfg.WITH_NORM,
            share_param=head_cfg.SHARE_PARAM,
        )
        self.num_classes = self.head.num_classes
        if self.matching == "sinkhorn":
            self.matcher = layers.SinkhornMatcher(eps=0.1, max_iter=50)
        elif self.matching == "topk":
            candidate_k = cfg.MODEL.HEAD.get("CANDIDATE_K", 10)
            self.matcher = layers.OTATopkMatcher(candidate_k=candidate_k)

    def network_forward(self, image):
        features = self.backbone(image)
        features = [features[f] for f in self.in_features]

        box_logits, box_offsets, box_iou = self.head(features)
        anchors_list = self.anchor_generator(features)

        box_logits_list = [permute_to_N_Any_K(x, K=self.num_classes) for x in box_logits]
        box_offsets_list = [permute_to_N_Any_K(x, K=4) for x in box_offsets]
        box_iou_list = [permute_to_N_Any_K(x, K=1) for x in box_iou]

        return box_logits_list, box_offsets_list, box_iou_list, anchors_list

    def get_losses(self, inputs):
        assert self.training

        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])
        box_cls, box_delta, box_iou, anchors = outputs
        gt_classes, gt_offsets, gt_ious = self.get_ground_truth(
            anchors, box_cls, box_delta, box_iou, inputs
        )
        losses = self.emd_losses(
            gt_classes, gt_offsets, gt_ious, box_cls, box_delta, box_iou
        )
        return losses

    def get_ground_truth(self, shifts, box_cls, box_delta, box_iou, inputs):
        gt_classes, gt_shifts_deltas, gt_ious = [], [], []

        box_cls = F.concat(box_cls, axis=1)
        box_delta = F.concat(box_delta, axis=1)
        box_iou = F.concat(box_iou, axis=1)
        all_shifts = F.concat(shifts, axis=0)

        for img_id, (img_box_cls, img_box_delta, img_box_iou) in enumerate(
            zip(box_cls, box_delta, box_iou)
        ):
            num_boxes = int(inputs["img_info"][img_id, -1].astype("int32"))
            # gt_boxes of shape (N, 5), last dim is boxes label
            gt_boxes = inputs["gt_boxes"][img_id, :num_boxes]

            # in gt box and center
            deltas = self.box_coder.encode(all_shifts, F.expand_dims(gt_boxes[:, :4], axis=1))
            is_in_boxes = deltas.min(axis=-1) > 0.01

            center_sampling_radius = 2.5
            # centers = gt_boxes.get_centers()
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2
            is_in_centers = []
            for stride, level_shifts in zip(self.head.strides, shifts):
                radius = stride * center_sampling_radius
                center_boxes = F.concat((
                    F.maximum(gt_centers - radius, gt_boxes[:, :2]),
                    F.minimum(gt_centers + radius, gt_boxes[:, 2:4]),
                ), axis=-1)
                center_deltas = self.box_coder.encode(
                    level_shifts, F.expand_dims(center_boxes, axis=1)
                )
                is_in_centers.append(center_deltas.min(axis=-1) > 0)
            is_in_centers = F.concat(is_in_centers, axis=1)
            del gt_centers, center_boxes, deltas, center_deltas
            is_in_boxes &= is_in_centers

            # NOTE: minus 1 in basedet because 0 is used for background in class label
            gt_cls_per_image = F.one_hot(
                gt_boxes[:, 4].astype("int32") - 1, self.head.num_classes
            ).astype("float32")

            num_anchor = len(all_shifts)
            shape = (num_boxes, num_anchor, self.head.num_classes)
            # no_grad
            loss_cls = layers.sigmoid_focal_loss(
                F.broadcast_to(F.expand_dims(img_box_cls, axis=0), shape),
                F.broadcast_to(F.expand_dims(gt_cls_per_image, axis=1), shape),
                alpha=self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA,
                gamma=self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA,
            ).sum(axis=-1).detach()

            loss_cls_bg = layers.sigmoid_focal_loss(
                img_box_cls,
                F.zeros_like(img_box_cls),
                alpha=self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA,
                gamma=self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA,
            ).sum(axis=-1).detach()

            gt_delta_per_image = self.box_coder.encode(
                all_shifts, F.expand_dims(gt_boxes[:, :4], axis=1)
            )

            delta_shape = (num_boxes, num_anchor, 4)
            loss_delta, ious = layers.iou_loss(
                F.broadcast_to(F.expand_dims(img_box_delta, axis=0), delta_shape),
                gt_delta_per_image,
                box_mode="ltrb",
                loss_type="iou",
                eps=np.finfo(np.float32).eps,
                return_iou=True,
            )
            loss_delta = loss_delta.detach()

            cost = loss_cls + self.reg_weight * loss_delta + 1e6 * (~is_in_boxes)

            if self.matching == "sinkhorn":
                cost = F.concat((cost, F.expand_dims(loss_cls_bg, axis=0)), axis=0)
                ious = ious * is_in_boxes.astype("float32")

            matched_gt_inds = self.matcher(cost, ious)

            gt_classes_i = F.zeros(num_anchor)  # zero means background class
            fg_mask = matched_gt_inds != num_boxes
            gt_classes_i[fg_mask] = gt_boxes[:, 4][matched_gt_inds[fg_mask]]
            gt_classes.append(gt_classes_i)

            box_target_per_image = F.zeros((num_anchor, 4))
            box_target_per_image[fg_mask] = gt_delta_per_image[
                matched_gt_inds[fg_mask], F.arange(num_anchor, dtype="int32")[fg_mask]
            ]
            gt_shifts_deltas.append(box_target_per_image)

            gt_ious_per_image = F.zeros((num_anchor, 1))
            gt_ious_per_image[fg_mask] = F.expand_dims(
                ious[matched_gt_inds[fg_mask], F.arange(num_anchor, dtype="int32")[fg_mask]],
                axis=1
            )
            gt_ious.append(gt_ious_per_image)
            # end of no_grad

        return (
            F.concat(gt_classes).detach(),
            F.concat(gt_shifts_deltas).detach(),
            F.concat(gt_ious).detach(),
        )

    def emd_losses(
        self, gt_classes, gt_shifts_deltas, gt_ious, pred_class_logits, pred_shift_deltas, pred_ious
    ):
        pred_class_logits = F.concat(pred_class_logits, axis=1).reshape(-1, self.num_classes)
        pred_shift_deltas = F.concat(pred_shift_deltas, axis=1).reshape(-1, 4)
        pred_ious = F.concat(pred_ious, axis=1).reshape(-1, 1)

        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.reshape(-1, 4)
        gt_ious = gt_ious.reshape(-1, 1)

        foreground_idxs = gt_classes > 0

        num_foreground = foreground_idxs.sum()
        gt_classes_target = F.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs].astype("int32") - 1] = 1

        num_foreground = all_reduce(num_foreground, mode="mean")

        # logits loss
        loss_cls = layers.sigmoid_focal_loss(
            pred_class_logits,
            gt_classes_target,
            alpha=self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA,
            gamma=self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA,
        ).sum() / max(1, num_foreground)

        # regression loss
        loss_box_reg = layers.iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.cfg.MODEL.LOSSES.IOU_LOSS_TYPE,
        ).sum() / max(1, num_foreground)

        # ious loss
        loss_ious = layers.binary_cross_entropy(
            pred_ious[foreground_idxs],
            gt_ious[foreground_idxs],
            with_logits=True,
        ).sum() / max(1, num_foreground)

        loss_box_reg *= 2.0
        loss_ious *= 0.5
        total_loss = loss_cls + loss_box_reg + loss_ious

        loss_dict = {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_offsets": loss_box_reg,
            "loss_ious": loss_ious,
        }
        return loss_dict

    def inference(self, inputs):
        # currently not support multi-batch testing
        assert not self.training

        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])
        box_logits_list, box_offsets_list, box_iou_list, anchors_list = outputs

        total_scores, logits_label, total_boxes = [], [], []

        for logits, offsets, centerness, anchors in zip(
            box_logits_list, box_offsets_list, box_iou_list, anchors_list,
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
