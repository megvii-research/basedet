#!/usr/bin/python3
# Copyright (c) Megvii, Inc. All rights reserved.

import megengine.functional as F

from basedet import layers
from basedet.layers import safelog
from basedet.structures import Boxes
from basedet.utils import registers

from .retinanet import RetinaNet


@registers.models.register()
class FreeAnchor(RetinaNet):

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.matcher  # FreeAnchor use bags to match

    def get_losses(self, inputs):
        assert self.training

        # get output meta data
        inputs = self.pre_process(inputs)
        box_logits_list, box_offsets_list, anchors_list = self.network_forward(inputs["image"])

        pred_logits = F.concat(box_logits_list, axis=1)
        pred_offsets = F.concat(box_offsets_list, axis=1)
        anchors = F.concat(anchors_list, axis=0)

        def positive_bag_loss(logits, axis=1):
            weight = 1.0 / (1.0 - logits)
            weight /= weight.sum(axis=axis, keepdims=True)
            bag_prob = (weight * logits).sum(axis=1)
            return -safelog(bag_prob)

        def negative_bag_loss(logits, gamma):
            return (logits ** gamma) * (-safelog(1.0 - logits))

        pred_scores = F.sigmoid(pred_logits)
        box_prob_list = []
        positive_losses = []
        clamp_eps = 1e-7
        bucket_size = self.cfg.MODEL.BUCKET.BUCKET_SIZE

        im_info = inputs["img_info"]
        gt_boxes = inputs["gt_boxes"]
        for batch_id, (gt_boxes_per_image, info_per_image) in enumerate(zip(gt_boxes, im_info)):
            num_instance = info_per_image[4].astype("int32")
            boxes_info = gt_boxes_per_image[:num_instance]
            # id 0 is used for background classes, so -1 first
            labels = boxes_info[:, 4].astype("int32") - 1

            # nograd begins
            pred_box = self.box_coder.decode(anchors, pred_offsets[batch_id]).detach()
            gt = Boxes(boxes_info[:, :4])
            overlaps = gt.iou(pred_box).detach()
            thresh1 = self.cfg.MODEL.BUCKET.BOX_IOU_THRESH
            thresh2 = F.clip(
                overlaps.max(axis=1, keepdims=True),
                lower=thresh1 + clamp_eps,
                upper=1.0
            )
            gt_pred_prob = F.clip(
                (overlaps - thresh1) / (thresh2 - thresh1), lower=0, upper=1.0)

            # guarantee that nonzero_idx is not empty
            # TODO: remove the workaround code
            fill_prob = False
            if gt_pred_prob.max() <= clamp_eps:
                fill_prob = True
                gt_pred_prob[0, 0] = 0.001

            _, nonzero_idx = F.cond_take(gt_pred_prob != 0, gt_pred_prob)
            # since nonzeros is only 1 dim, use num_anchor to get real indices
            num_anchors = gt_pred_prob.shape[1]

            anchors_idx = nonzero_idx % num_anchors
            gt_idx = nonzero_idx // num_anchors

            image_boxes_prob = F.zeros(pred_logits.shape[1:]).detach()
            image_boxes_prob[anchors_idx, labels[gt_idx]] = gt_pred_prob[gt_idx, anchors_idx]
            # remove effect of setting gt_pred_prob
            if fill_prob:
                image_boxes_prob[0, 0] = 0.0

            box_prob_list.append(image_boxes_prob)
            # nograd end

            # construct bags for objects
            match_quality_matrix = gt.iou(anchors).detach()
            num_gt = match_quality_matrix.shape[0]
            _, matched_idx = F.topk(
                match_quality_matrix, k=bucket_size, descending=True, no_sort=True,
            )
            matched_idx = matched_idx.detach()
            matched_idx_flatten = matched_idx.reshape(-1)

            gather_idx = labels.reshape(-1, 1, 1)
            gather_idx = F.broadcast_to(gather_idx, (num_gt, bucket_size, 1))

            gather_src = pred_scores[batch_id, matched_idx_flatten]
            gather_src = gather_src.reshape(num_gt, bucket_size, -1)
            matched_score = F.gather(gather_src, 2, gather_idx)
            matched_score = F.squeeze(matched_score, axis=2)

            topk_anchors = anchors[matched_idx_flatten]
            boxes_broad_cast = F.broadcast_to(
                F.expand_dims(gt, axis=1), (num_gt, bucket_size, 4)
            ).reshape(-1, 4)
            matched_offsets = self.box_coder.encode(
                topk_anchors,
                boxes_broad_cast
            )

            reg_loss = layers.smooth_l1_loss(
                pred_offsets[batch_id, matched_idx_flatten],
                matched_offsets,
                beta=self.cfg.MODEL.LOSSES.SMOOTH_L1_BETA
            ).sum(axis=-1) * self.cfg.MODEL.LOSSES.REG_LOSS_WEIGHT
            matched_reg_scores = F.exp(-reg_loss)

            positive_losses.append(
                positive_bag_loss(
                    matched_score * matched_reg_scores.reshape(-1, bucket_size),
                    axis=1
                )
            )

        num_foreground = im_info[:, 4].sum()
        pos_loss = F.concat(positive_losses).sum() / F.maximum(1.0, num_foreground)
        box_probs = F.stack(box_prob_list, axis=0)

        neg_loss = negative_bag_loss(
            pred_scores * (1 - box_probs), self.cfg.MODEL.LOSSES.FOCAL_LOSS_GAMMA
        ).sum() / F.maximum(1.0, num_foreground * bucket_size)

        alpha = self.cfg.MODEL.LOSSES.FOCAL_LOSS_ALPHA
        pos_loss = pos_loss * alpha
        neg_loss = neg_loss * (1 - alpha)
        loss_dict = {
            "total_loss": pos_loss + neg_loss,
            "pos_loss": pos_loss,
            "neg_loss": neg_loss,
        }
        return loss_dict

    def get_ground_truth(self, anchors, batched_gt_boxes, batched_valid_gt_box_number):
        """since FreeAnchor use bags to match, get_ground_truth is not implemented"""
        raise NotImplementedError
