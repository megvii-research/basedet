#!/usr/bin/python3
# Copyright (C) 2019-2021 Megvii Inc. All rights reserved.

import numpy as np

import megengine as mge
import megengine.functional as F

import basedet.models.cls as cls_net
from basedet import layers
from basedet.models.base_net import BaseNet
from basedet.structures import BoxConverter, Boxes, Container
from basedet.utils import registers


@registers.models.register()
class YOLOv3(BaseNet):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.DATA.NUM_CLASSES
        model_cfg = cfg.MODEL
        self.batch_size = model_cfg.BATCHSIZE

        # build backbone
        bottom_up = getattr(cls_net, model_cfg.BACKBONE.NAME)(
            out_features=model_cfg.BACKBONE.OUT_FEATURES,
        )
        anchors = model_cfg.ANCHOR.SCALES
        self.anchors = anchors

        self.backbone = layers.YOLOFPN(
            bottom_up=bottom_up,
            in_features=model_cfg.BACKBONE.OUT_FEATURES,
            out_channels=[len(x) * (5 + self.num_classes) for x in anchors],
        )

        anchors_array = np.array(anchors).reshape(-1, 2)
        all_anchors = np.zeros((len(anchors_array), 4))
        all_anchors[:, 2:] = anchors_array
        self.all_anchors = mge.tensor(all_anchors)

        # build head
        self.head = layers.YOLOHead(
            output_dim=5 + self.num_classes,
            num_anchors=[len(x) for x in anchors],
        )

        self.mean = mge.tensor(model_cfg.BACKBONE.IMG_MEAN).reshape(1, -1, 1, 1)
        self.std = mge.tensor(model_cfg.BACKBONE.IMG_STD).reshape(1, -1, 1, 1)

        # TODO refine the following attribute
        # self.conf_threshold = cfg.MODEL.CONF_THRESHOLD
        # self.nms_threshold = cfg.MODEL.NMS_THRESHOLD
        # self.nms_type = cfg.MODEL.NMS_TYPE
        self.ignore_threshold = model_cfg.IGNORE_THRESHOLD

        self.target_size = 512
        self.multi_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        self.change_iter = 10
        self.iter = 0
        self.max_iter = 100000  # caculate

    def pre_process(self, inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = inputs.pop("data") if isinstance(inputs, dict) else inputs
        # image is rectangle_input
        origin_size = images.shape[-1]

        # padded_image = layers.get_padded_tensor(image, 32, 0.0)
        images: mge.Tensor = layers.data_to_input(images / 255.0, self.mean, self.std)

        if self.training:
            # resize image inputs
            mode = np.random.choice(["bilinear", "nearest", "bicubic"])
            upsample_kwargs = {}
            if mode == "bilinear":
                upsample_kwargs = {"align_corners": False}
            images = F.vision.interpolate(
                images, size=[self.target_size, self.target_size], mode=mode, **upsample_kwargs,
            )

            # resize boxes also
            boxes = inputs["gt_boxes"]
            boxes[..., :4] *= self.target_size / origin_size
        else:
            self.target_size = 608
            images = F.vision.interpolate(
                images, size=[self.target_size, self.target_size],
                mode="bilinear", align_corners=False,
            )
            inputs["im_info"][:, :2] = [self.target_size, self.target_size]

        inputs["image"] = images
        return inputs

    def network_forward(self, image):
        features = self.backbone(image)
        outputs = self.head(features)
        return outputs

    def get_losses(self, inputs):
        assert self.training
        # get output meta data
        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])

        loss_list = []
        for level, (pred, anchor) in enumerate(zip(outputs, self.anchors)):
            target_shape = pred.shape[:4]
            batch_size, _, in_h, in_w = target_shape

            stride = self.target_size / in_w, self.target_size / in_h
            pred_boxes, x, y, w, h = self.decode_pred_boxes(
                pred, anchor, target_shape, stride
            )
            conf = F.sigmoid(pred[..., 4])       # box confidence
            pred_cls = F.sigmoid(pred[..., 5:])  # class prediction

            #  build target
            mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls = self.get_ground_truth(
                inputs, pred_boxes, target_shape, stride, level
            )

            # TODO check and refine the loss logic
            loss_x = (
                mask * tgt_scale * F.nn.binary_cross_entropy(x, tx, "none")
            ).sum() / self.batch_size
            loss_y = (
                mask * tgt_scale * F.nn.binary_cross_entropy(y * mask, ty * mask, "none")
            ).sum() / self.batch_size
            loss_w = (
                mask * tgt_scale * F.nn.l1_loss(w * mask, tw * mask, "none")
            ).sum() / self.batch_size
            loss_h = (
                mask * tgt_scale * F.nn.l1_loss(h * mask, th * mask, "none")
            ).sum() / self.batch_size

            loss_conf = (
                obj_mask * F.nn.binary_cross_entropy(conf, mask, "none")
            ).sum() / self.batch_size
            loss_cls = (
                F.nn.binary_cross_entropy(pred_cls, tcls, "none")
            ).sum() / self.batch_size

            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            loss = {
                "total_loss": total_loss,
                "loss_x": loss_x,
                "loss_y": loss_y,
                "loss_w": loss_w,
                "loss_h": loss_h,
                "loss_conf": loss_conf,
                "loss_cls": loss_cls,
            }
            loss_list.append(loss)

        loss_keys = loss_list[0].keys()
        loss_dict = {k: sum([loss[k] for loss in loss_list]) for k in loss_keys}
        return loss_dict

    def inference(self, inputs):
        assert not self.training
        # currently not support multi-batch testing during inference
        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"])

        total_scores, logits_label, total_boxes = [], [], []
        for pred, anchor in zip(outputs, self.anchors):
            target_shape = pred.shape[:4]
            in_h, in_w = target_shape[-2:]

            stride = self.target_size / in_w, self.target_size / in_h
            pred_boxes = self.decode_pred_boxes(pred, anchor, target_shape, stride).reshape(-1, 4)
            box_conf = F.sigmoid(pred[..., 4]).reshape(-1, 1)       # box confidence
            logits = F.sigmoid(pred[..., 5:]).reshape(-1, self.num_classes)  # class prediction
            scores = F.flatten(box_conf * logits)

            _, keep_idx = layers.non_zeros(scores > self.cfg.TEST.CLS_THRESHOLD)
            if layers.is_empty_tensor(keep_idx):
                continue
            topk_num = min(keep_idx.shape[0], 1000)
            _, topk_idx = F.topk(scores[keep_idx], k=topk_num, descending=True)
            keep_idx = keep_idx[topk_idx]

            total_scores.append(scores[keep_idx])
            logits_label.append(keep_idx % self.num_classes)
            total_boxes.append(pred_boxes[keep_idx // self.num_classes])

        if not total_boxes:
            empty_tensor = mge.Tensor([])
            return Container(boxes=empty_tensor, box_scores=empty_tensor, box_labels=empty_tensor)

        boxes_container = Container(
            boxes=Boxes(
                BoxConverter.convert(F.concat(total_boxes, axis=0), "XcYcWH2XYXY")
            ),
            box_scores=F.concat(total_scores, axis=0),
            box_labels=F.concat(logits_label, axis=0),
        )

        processed_boxes = layers.post_processing(
            boxes_container,
            inputs["im_info"],
            iou_threshold=self.cfg.TEST.IOU_THRESHOLD,
            max_detections_per_image=self.cfg.TEST.MAX_BOXES_PER_IMAGE,
        )
        return processed_boxes

    def decode_pred_boxes(self, pred, anchor, target_shape, stride):
        batch_size, num_anchor, in_h, in_w = target_shape
        stride_w, stride_h = stride

        grid_x = F.broadcast_to(
            F.arange(in_w),
            shape=(batch_size * num_anchor, in_h, in_w)
        ).reshape(target_shape)
        grid_y = F.broadcast_to(
            F.broadcast_to(F.arange(in_h), (in_w, in_h)).transpose(),
            shape=(batch_size * num_anchor, in_h, in_w),
        ).reshape(target_shape)

        # Calculate anchor w, h
        anchor_w = mge.tensor(anchor)[..., 0].reshape(1, -1, 1, 1)
        anchor_h = mge.tensor(anchor)[..., 1].reshape(1, -1, 1, 1)

        anchor_w = F.broadcast_to(anchor_w, target_shape)
        anchor_h = F.broadcast_to(anchor_h, target_shape)

        x = F.sigmoid(pred[..., 0])  # x center
        y = F.sigmoid(pred[..., 1])  # y center
        w = pred[..., 2]  # box width
        h = pred[..., 3]  # box height

        # Add offset and scale with anchors
        pred_boxes = pred[..., :4].detach()
        pred_boxes[..., 0] = (x + grid_x) * stride_w
        pred_boxes[..., 1] = (y + grid_y) * stride_h
        pred_boxes[..., 2] = F.exp(w) * anchor_w
        pred_boxes[..., 3] = F.exp(h) * anchor_h
        if self.training:
            return pred_boxes, w, y, w, h
        else:
            return pred_boxes

    def get_ground_truth(self, inputs, pred_boxes, target_shape, stride, level, eps=1e-16):
        stride_w, stride_h = stride

        mask, obj_mask = F.zeros(target_shape), F.ones(target_shape) > 0
        tx, ty = F.zeros(target_shape), F.zeros(target_shape)
        tw, th = F.zeros(target_shape), F.zeros(target_shape)
        tgt_scale, tcls = F.zeros(target_shape), F.zeros(target_shape + (self.num_classes,))

        gt_boxes, boxes_info = inputs["gt_boxes"], inputs["im_info"]
        gt_boxes[..., -1] -= 1  # minus 1 for coco datasets to avoid illegal mem access.
        gt_x_center = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.0
        gt_y_center = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.0
        gt_width = (gt_boxes[..., 2] - gt_boxes[..., 0])
        gt_height = (gt_boxes[..., 3] - gt_boxes[..., 1])
        gt_x_index = (gt_x_center / stride_w).astype("int32")
        gt_y_index = (gt_y_center / stride_h).astype("int32")

        num_boxes = boxes_info[..., -1]
        for bid in range(target_shape[0]):
            num_valid_boxes = int(num_boxes[bid])
            if num_valid_boxes == 0:
                continue

            truth_box = F.zeros((num_valid_boxes, 4))
            truth_box[:, 2] = gt_width[bid, :num_valid_boxes]
            truth_box[:, 3] = gt_height[bid, :num_valid_boxes]
            truth_i = gt_x_index[bid, :num_valid_boxes]
            truth_j = gt_y_index[bid, :num_valid_boxes]

            anchor_ious_all = bboxes_iou(truth_box, self.all_anchors)
            best_n_all = F.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all // 3) == level)

            truth_box[:num_valid_boxes, 0] = gt_x_center[bid, :num_valid_boxes]
            truth_box[:num_valid_boxes, 1] = gt_y_center[bid, :num_valid_boxes]
            pred_box = pred_boxes[bid]

            pred_ious = bboxes_iou(pred_box.reshape(-1, 4), truth_box)

            pred_best_iou = pred_ious.max(axis=1)
            pred_best_iou = (pred_best_iou > self.ignore_threshold)
            pred_best_iou = pred_best_iou.reshape(pred_box.shape[:3])
            obj_mask[bid] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for t in range(best_n.shape[0]):
                if best_n_mask[t] == 1:
                    gi, gj = truth_i[t], truth_j[t]
                    gx, gy = gt_x_center[bid, t], gt_y_center[bid, t]
                    gw, gh = gt_width[bid, t], gt_height[bid, t]

                    a = best_n[t]

                    mask[bid, a, gj, gi] = 1
                    obj_mask[bid, a, gj, gi] = 1

                    # Coordinates
                    tx[bid, a, gj, gi] = gx / stride_w - gi
                    ty[bid, a, gj, gi] = gy / stride_h - gj
                    # Width and height
                    tw[bid, a, gj, gi] = F.log((gw / self.anchors[level][a][0] + eps).astype("float32"))  # noqa
                    th[bid, a, gj, gi] = F.log((gh / self.anchors[level][a][1] + eps).astype("float32"))  # noqa

                    tgt_scale[bid, a, gj, gi] = 2.0 - gw * gh / (self.target_size * self.target_size)  # noqa
                    # One-hot encoding of label
                    tcls[bid, a, gj, gi, int(gt_boxes[bid, t, -1])] = 1

        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls


def bboxes_iou(bboxes_a, bboxes_b):
    box_a = BoxConverter.convert(bboxes_a, "XcYcWH2XYXY")
    box_b = BoxConverter.convert(bboxes_b, "XcYcWH2XYXY")
    return Boxes(box_a).iou(box_b)
