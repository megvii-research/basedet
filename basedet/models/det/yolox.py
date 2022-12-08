#!/usr/bin/env python3

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M

from basedet import layers
from basedet.models import BaseNet
from basedet.structures import BoxConverter
from basedet.utils import registers

from ..cls.csp_darknet import csp_darknet


@registers.models.register()
class YOLOX(BaseNet):
    """
    YOLOX model module. The network returns loss values from three YOLO layers
    during training and detection results during test.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.MODEL
        self.depth_factor = cfg.MODEL.DEPTH_FACTOR
        self.width_factor = cfg.MODEL.WIDTH_FACTOR
        self.batch_size = model_cfg.BATCHSIZE
        self.num_anchors = 1
        self.conf_thresh = 0.7
        self.use_l1 = False
        self.strides = [8, 16, 32]
        self.bn_eps = model_cfg.BN_EPS
        self.bn_momentum = model_cfg.BN_MOMENTUM
        self.target_size = cfg.AUG.TRAIN_SETTING.INPUT_SIZE

        # build yolox backbone
        if model_cfg.BACKBONE.NAME == "csp_darknet":
            bottom_up = csp_darknet(
                self.depth_factor, self.width_factor,
                depthwise=model_cfg.DEPTHWISE,
                out_features=model_cfg.BACKBONE.OUT_FEATURES,
                activation=model_cfg.ACTIVATION,
            )
        else:
            bottom_up = layers.build_backbone(model_cfg.BACKBONE)
        self.backbone = layers.YOLOPAFPN(
            bottom_up, depth=self.depth_factor, width=self.width_factor,
        )
        self.anchor_generator = layers.FastPointGenerator(self.strides)

        in_channels = [int(x * self.width_factor) for x in [256, 512, 1024]]
        mid_channels = int(256 * self.width_factor)
        self.head = layers.YOLOXHead(
            cfg.DATA.NUM_CLASSES,
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.grids = [np.array(0)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)
        self.set_model_hyperparam()

    def set_model_hyperparam(self):
        for module in self.modules():
            if isinstance(module, M.BatchNorm2d):
                module.eps = self.bn_eps
                module.momentum = self.bn_momentum

    def pre_process(self, inputs):
        images = mge.Tensor(inputs["data"]) if isinstance(inputs, dict) else inputs
        processed_data = {"image": images}

        if self.training:
            # resize image size during training
            gt_boxes = mge.Tensor(inputs["gt_boxes"])
            h, w = images.shape[-2:]
            if (h, w) != self.target_size:
                scale_h, scale_w = self.target_size[0] / h, self.target_size[1] / w
                images = F.vision.interpolate(
                    images, size=self.target_size, mode="bilinear", align_corners=False
                )
                gt_boxes[..., 0:4:2] = gt_boxes.detach()[..., 0:4:2] * scale_w
                gt_boxes[..., 1:4:2] = gt_boxes.detach()[..., 1:4:2] * scale_h
                # gt_boxes[..., 0:4:2] *= scale_w
                # gt_boxes[..., 1:4:2] *= scale_h
            processed_data["gt_boxes"] = gt_boxes
            processed_data["image"] = images

        if not isinstance(inputs, dict) or "im_info" not in inputs:
            h, w = images.shape
            img_info = mge.Tensor([[h, w, h, w]])
        else:
            img_info = mge.Tensor(inputs["im_info"])
        processed_data.update(img_info=img_info)

        return processed_data

    def network_forward(self, image):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(image)
        logits, offsets, objs = self.head(fpn_outs)
        all_anchors = self.anchor_generator(logits)
        return logits, offsets, objs, all_anchors

    def inference(self, inputs):
        assert not self.training
        inputs = self.pre_process(inputs)
        logits, offsets, objs, all_anchors = self.network_forward(inputs["image"])

        num_classes = self.head.num_classes
        logits = [layers.permute_to_N_Any_K(x, K=num_classes) for x in logits]
        objs = [layers.permute_to_N_Any_K(x, K=1) for x in objs]
        offsets = [layers.permute_to_N_Any_K(x, K=4) for x in offsets]
        scores = [F.sqrt(F.sigmoid(x) * F.sigmoid(y)).flatten() for x, y in zip(logits, objs)]
        del objs, logits

        total_scores, logits_label, total_boxes = [], [], []
        for anchor, coord, score, stride in zip(all_anchors, offsets, scores, self.strides):
            coord[..., :2] = coord[..., :2] * stride + anchor  # x, y
            coord[..., 2:] = stride * F.exp(coord[..., 2:])  # w, h

            boxes = BoxConverter.convert(coord.reshape(-1, 4), "xcycwh2xyxy")

            _, keep_idx = layers.non_zeros(score > self.cfg.TEST.CLS_THRESHOLD)
            if layers.is_empty_tensor(keep_idx):
                continue
            topk_num = min(keep_idx.shape[0], 1000)
            _, topk_idx = F.topk(score[keep_idx], k=topk_num, descending=True)
            keep_idx = keep_idx[topk_idx]

            total_scores.append(score[keep_idx])
            logits_label.append(keep_idx % num_classes)
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

    def get_losses(self, inputs):
        assert self.training
        inputs = self.pre_process(inputs)
        logits, offsets, objs, all_anchors = self.network_forward(inputs["image"])

        logits = [layers.permute_to_N_Any_K(x, K=self.head.num_classes) for x in logits]
        objs = [layers.permute_to_N_Any_K(x, K=1) for x in objs]
        offsets = [layers.permute_to_N_Any_K(x, K=4) for x in offsets]
        if self.use_l1:
            origin_offsets = [mge.Tensor(x) for x in offsets]

        # outputs = []
        for anchor, coord, stride in zip(all_anchors, offsets, self.strides):
            coord[..., :2] = coord[..., :2] * stride + anchor  # Xc, Yc
            coord[..., 2:] = stride * F.exp(coord[..., 2:])  # w, h
            # output = F.concat([coord, obj, logit], axis=-1)
            # outputs.append(output)

        # outputs : decoded results
        # x_shifts, y_shifts : not multi stride
        # expanded_stride: strdie of different level
        bbox_preds = F.concat(offsets, axis=1)
        obj_preds = F.concat(objs, axis=1)
        cls_preds = F.concat(logits, axis=1)

        cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []
        num_fg = 0.0
        num_classes = self.head.num_classes

        labels = inputs["gt_boxes"]
        num_objects = (labels[..., -1] > 0).sum(axis=-1)

        batch_size, total_num_anchors = bbox_preds.shape[:2]
        for batch_idx in range(batch_size):
            num_gt = int(num_objects[batch_idx])
            if num_gt == 0:
                cls_target = F.zeros((0, num_classes))
                reg_target = F.zeros((0, 4))
                l1_target = F.zeros((0, 4))
                obj_target = F.zeros((total_num_anchors, 1))
                fg_mask = F.zeros(total_num_anchors).astype("bool")
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]
                # gt_classes should mius 1 since COCO start at 1 in basedet
                gt_classes = labels[batch_idx, :num_gt, 4] - 1

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds = self.get_assignments(  # noqa
                    gt_bboxes_per_image,
                    gt_classes,
                    bbox_preds[batch_idx],
                    cls_preds[batch_idx],
                    obj_preds[batch_idx],
                    all_anchors,
                )
                num_fg_img = fg_mask.sum()

                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.astype("int32"), num_classes
                ) * F.expand_dims(pred_ious_this_matching, axis=-1)
                obj_target = F.expand_dims(fg_mask, axis=-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        gt_bboxes_per_image[matched_gt_inds], all_anchors, fg_mask
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = F.concat(cls_targets, 0)
        reg_targets = F.concat(reg_targets, 0)
        obj_targets = F.concat(obj_targets, 0)
        fg_masks = F.concat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = layers.iou_loss(
            bbox_preds.reshape(-1, 4)[fg_masks], reg_targets,
            box_mode="xcycwh", loss_type="square_iou",
        )
        loss_iou = loss_iou[F.eye(loss_iou.shape[0]) == 1]  # use diag value of loss_iou
        loss_iou = loss_iou.sum() / num_fg

        loss_obj = layers.binary_cross_entropy(obj_preds.reshape(-1, 1), obj_targets).sum() / num_fg
        loss_cls = layers.binary_cross_entropy(
            cls_preds.reshape(-1, num_classes)[fg_masks], cls_targets
        ).sum() / num_fg

        if self.use_l1:
            l1_targets = F.concat(l1_targets, 0)
            origin_offsets = F.concat(origin_offsets, 1).reshape(-1, 4)
            l1_loss = layers.smooth_l1_loss(
                origin_offsets[fg_masks], l1_targets, beta=0.0,
            ).sum() / num_fg
        else:
            l1_loss = mge.Tensor(0.0)

        reg_weight = 5.0
        loss_iou *= reg_weight
        loss = loss_iou + loss_obj + loss_cls + l1_loss

        self.extra_meter.update({"img_size": self.target_size[0]})

        output_dict = {
            "total_loss": loss,
            "iou_loss": loss_iou,
            "l1_loss": l1_loss,
            "obj_loss": loss_obj,
            "cls_loss": loss_cls,
        }
        return output_dict

    def get_l1_target(self, gt, all_anchors, fg_mask, eps=1e-8):
        # logic of gt from xcycwh -> xywh
        shifts = F.concat(all_anchors, axis=0)[fg_mask]
        stride = F.concat(
            [F.full(x.shape[0], s) for (s, x) in zip(self.strides, all_anchors)]
        )[fg_mask]
        stride = F.expand_dims(stride, axis=-1)
        center_target = ((gt[:, :2] + gt[:, 2:]) / 2 - shifts) / stride
        wh_target = F.log((gt[:, 2:] - gt[:, :2]) / stride + eps)
        return F.concat([center_target, wh_target], axis=-1)

    @classmethod
    def tlbr_iou(cls, boxes1, boxes2):
        tl = F.maximum(
            F.expand_dims(boxes1[:, :2] - boxes1[:, 2:] / 2, axis=1),
            (boxes2[:, :2] - boxes2[:, 2:] / 2),
        )
        br = F.minimum(
            F.expand_dims(boxes1[:, :2] + boxes1[:, 2:] / 2, axis=1),
            (boxes2[:, :2] + boxes2[:, 2:] / 2),
        )
        area_a = boxes1[:, 2] * boxes1[:, 3]
        area_b = boxes2[:, 2] * boxes2[:, 3]
        mask = (tl < br).astype("int32")
        mask = mask[..., 0] * mask[..., 1]
        diff = br - tl
        area_i = diff[..., 0] * diff[..., 1] * mask
        return area_i / (F.expand_dims(area_a, axis=1) + area_b - area_i)

    def get_assignments(
        self, gt_bboxes_per_image, gt_classes,
        bboxes_preds, cls_preds, obj_preds, all_anchors,
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, all_anchors)
        bboxes_preds = bboxes_preds[fg_mask]
        cls_preds_ = cls_preds[fg_mask]
        obj_preds_ = obj_preds[fg_mask]
        pair_wise_ious = self.tlbr_iou(gt_bboxes_per_image, bboxes_preds)
        num_in_boxes_anchor = bboxes_preds.shape[0]

        # MGE might bring bad exper
        # caused by wrong boxes value
        gt_cls_per_image = (
            F.repeat(
                F.expand_dims(
                    F.one_hot(gt_classes.astype("int32"), self.head.num_classes).astype("float32"),
                    axis=1,
                ),
                repeats=num_in_boxes_anchor, axis=1,
            )
        )
        pair_wise_ious_loss = -F.log(pair_wise_ious + 1e-8)

        num_gt = gt_bboxes_per_image.shape[0]
        # ditto
        cls_preds_ = F.repeat(
            F.expand_dims(F.sigmoid(cls_preds_) * F.sigmoid(obj_preds_), axis=0),
            repeats=num_gt, axis=0,
        )
        pair_wise_cls_loss = layers.binary_cross_entropy(
            F.sqrt(cls_preds_), gt_cls_per_image, with_logits=False,
        ).sum(-1)
        del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)
        return self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

    def get_in_boxes_info(self, gt_bboxes_per_image, all_anchors):
        num_gt = gt_bboxes_per_image.shape[0]
        # TODO NOTE anchors offset 0.5
        anchors = F.concat(all_anchors, axis=0)
        num_anchors = anchors.shape[0]
        grids = (
            F.repeat(
                F.expand_dims(anchors, axis=0), repeats=num_gt, axis=0,
            )
        )  # [n_anchor] -> [n_gt, n_anchor]

        x1y1 = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, :2], axis=1), repeats=num_anchors, axis=1,
        )
        x2y2 = F.repeat(
            F.expand_dims(gt_bboxes_per_image[:, 2:4], axis=1), repeats=num_anchors, axis=1,
        )

        is_in_boxes = F.minimum(grids - x1y1, x2y2 - grids).min(axis=-1) > 0.0
        is_in_boxes_all = is_in_boxes.sum(axis=0) > 0
        # in fixed center

        xcyc = (x1y1 + x2y2) / 2
        center_radius = 2.5
        all_strides = F.concat(
            [F.full(x.shape[0], s) for (s, x) in zip(self.strides, all_anchors)]
        ).reshape(1, -1, 1)

        tl = xcyc - center_radius * all_strides
        br = xcyc + center_radius * all_strides

        is_in_centers = F.minimum(grids - tl, br - grids).min(axis=-1) > 0.0
        is_in_centers_all = is_in_centers.sum(axis=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor.detach(), is_in_boxes_and_center.detach()

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # TODO OTA matching
        matching_matrix = F.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = F.topk(ious_in_boxes_matrix, n_candidate_k, descending=True)
        dynamic_ks = F.clip(topk_ious.sum(1).astype("int32"), lower=1)
        for gt_idx in range(num_gt):
            _, pos_idx = F.topk(cost[gt_idx], k=dynamic_ks[gt_idx], descending=False)
            matching_matrix[gt_idx, pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_argmin = F.argmin(cost[:, anchor_matching_gt > 1], axis=0)
            matching_matrix[:, anchor_matching_gt > 1] = 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0

        # set True part to fg_mask_inboxes
        fg_mask[fg_mask] = fg_mask_inboxes

        matched_gt_inds = F.argmax(matching_matrix[:, fg_mask_inboxes], axis=0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return (
            gt_matched_classes.detach(),
            fg_mask,
            pred_ious_this_matching.detach(),
            matched_gt_inds.detach(),
        )
