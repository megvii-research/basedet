#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. All rights reserved.

import megengine as mge

from basedet import layers
from basedet.models.base_net import BaseNet
from basedet.structures import Boxes, Container
from basedet.utils import registers

from .rpn import RPN


@registers.models.register()
class FasterRCNN(BaseNet):
    """
    Implement Faster R-CNN (https://arxiv.org/abs/1506.01497).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.MODEL

        # ----------- backbone ----------- #
        bottom_up = layers.build_backbone(model_cfg.BACKBONE)

        # ----------- FPN ---------------- #
        out_channels = 256
        self.backbone = layers.FPN(
            bottom_up=bottom_up,
            in_features=model_cfg.BACKBONE.OUT_FEATURES,
            out_channels=out_channels,
            norm=model_cfg.FPN.NORM,
            top_block=layers.FPNP6(model_cfg.FPN.TOP_BLOCK_IN_FEATURE),
            strides=[4, 8, 16, 32],
            channels=[256, 512, 1024, 2048],
        )

        # ---------- RPN and RCNN -------- #
        self.rpn = RPN(cfg)
        self.rcnn = layers.RCNN(cfg)
        self.img_mean = mge.tensor(self.cfg.MODEL.BACKBONE.IMG_MEAN).reshape(1, -1, 1, 1)
        self.img_std = mge.tensor(self.cfg.MODEL.BACKBONE.IMG_STD).reshape(1, -1, 1, 1)

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

    def network_forward(self, inputs):
        features = self.backbone(inputs["image"])

        if self.training:
            img_info = inputs["img_info"]
            gt_boxes = inputs["gt_boxes"]
            rpn_rois, rpn_losses = self.rpn(features, img_info, gt_boxes)
            rcnn_losses = self.rcnn(features, rpn_rois, img_info, gt_boxes)
            return rpn_losses, rcnn_losses
        else:
            img_info = inputs["img_info"]
            rpn_rois = self.rpn(features, img_info)
            pred_boxes, pred_score = self.rcnn(features, rpn_rois)
            return pred_boxes, pred_score

    def get_losses(self, inputs):
        assert self.training
        inputs = self.pre_process(inputs)
        rpn_losses, rcnn_losses = self.network_forward(inputs)
        loss_rpn_cls = rpn_losses["loss_rpn_cls"]
        loss_rpn_bbox = rpn_losses["loss_rpn_bbox"]
        loss_rcnn_cls = rcnn_losses["loss_rcnn_cls"]
        loss_rcnn_bbox = rcnn_losses["loss_rcnn_bbox"]
        total_loss = loss_rpn_cls + loss_rpn_bbox + loss_rcnn_cls + loss_rcnn_bbox

        loss_dict = {
            "total_loss": total_loss,
            "rpn_cls_loss": loss_rpn_cls,
            "rpn_reg_loss": loss_rpn_bbox,
            "rcnn_cls_loss": loss_rcnn_cls,
            "rcnn_reg_loss": loss_rcnn_bbox,
        }
        return loss_dict

    def inference(self, inputs):
        assert not self.training
        inputs = self.pre_process(inputs)
        pred_boxes, pred_scores = self.network_forward(inputs)
        return self.post_process(pred_boxes, pred_scores, inputs["img_info"])

    def post_process(self, boxes, scores, img_info):
        boxes = boxes.reshape(-1, 4)

        # TODO: 0.05 to config
        score_threshold = 0.05
        if scores.max() < score_threshold:
            # avoid empty tensor operation
            empty_tensor = mge.Tensor([])
            return Container(boxes=empty_tensor, box_scores=empty_tensor, box_labels=empty_tensor)

        _, box_index = layers.non_zeros(scores > score_threshold)
        boxes = boxes[box_index]
        scores = scores.reshape(-1)[box_index]
        # TODO consider class_awre_box
        labels = box_index % self.rcnn.num_classes

        processed_boxes = layers.post_processing(
            Container(boxes=Boxes(boxes), box_scores=scores, box_labels=labels),
            img_info,
            iou_threshold=self.cfg.TEST.IOU_THRESHOLD,
            max_detections_per_image=self.cfg.TEST.MAX_BOXES_PER_IMAGE,
        )
        return processed_boxes
