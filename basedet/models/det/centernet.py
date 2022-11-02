#!/usr/bin/env python3

import cv2
import numpy as np

import megengine as mge
import megengine.functional as F

from basedet import layers
from basedet.models.base_net import BaseNet
from basedet.structures import Boxes, Container
from basedet.utils import registers

__all__ = ["CenterNet"]


@registers.models.register()
class CenterNet(BaseNet):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.MODEL
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.tensor_dim = model_cfg.HEAD.TENSOR_DIM
        self.box_scale = 1 / model_cfg.HEAD.DOWN_SCALE
        self.output_size = model_cfg.OUTPUT_SIZE
        self.min_overlap = model_cfg.HEAD.MIN_OVERLAP

        self.backbone = layers.build_backbone(model_cfg.BACKBONE)

        self.upsample = layers.CenternetDeconv(
            channels=model_cfg.HEAD.DECONV_CHANNEL,
            deconv_kernel_sizes=model_cfg.HEAD.DECONV_KERNEL,
            modulate_deform=model_cfg.HEAD.MODULATE_DEFORM,
        )

        self.head = layers.CenterHead(
            in_channels=model_cfg.HEAD.IN_CHANNELS,
            num_classes=self.num_classes,
            prior_prob=model_cfg.HEAD.CLS_PRIOR_PROB,
        )

        img_mean = model_cfg.BACKBONE.IMG_MEAN
        if img_mean is not None:
            self.img_mean = mge.tensor(img_mean).reshape(1, -1, 1, 1)
        img_std = model_cfg.BACKBONE.IMG_STD
        if img_std is not None:
            self.img_std = mge.tensor(img_std).reshape(1, -1, 1, 1)

    def pre_process(self, inputs):
        """
        Normalize, pad and batch the input image.
        """
        image = super().pre_process(inputs["data"])
        processed_data = {"image": mge.Tensor(image)}
        if self.training:
            processed_data.update(gt_boxes=mge.Tensor(inputs["gt_boxes"]))

        if not isinstance(inputs, dict) or "im_info" not in inputs:
            h, w = image.shape
            img_info = mge.Tensor([[h, w, h, w]])
        else:
            img_info = mge.Tensor(inputs["im_info"])
        processed_data.update(img_info=img_info)
        return processed_data

    def network_forward(self, image):
        features = self.backbone.extract_features(image)["res5"]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)
        return pred_dict

    def get_ground_truth(self, batched_inputs):
        scoremap_list, wh_list, reg_list, reg_mask_list, index_list = [[] for i in range(5)]
        gt_boxes, img_info = batched_inputs["gt_boxes"], batched_inputs["img_info"]

        for gt_boxes_per_img, info_per_img in zip(gt_boxes, img_info):
            # init gt tensors
            gt_scoremap = F.zeros((self.num_classes, *self.output_size))
            gt_wh = F.zeros((self.tensor_dim, 2))
            gt_reg = F.zeros_like(gt_wh)
            reg_mask = F.zeros(self.tensor_dim)
            gt_index = F.zeros(self.tensor_dim).astype("int32")

            num_boxes = info_per_img[-1].astype("int32")
            gt_boxes_per_img = gt_boxes_per_img[:num_boxes]
            boxes = gt_boxes_per_img[:, :4]
            # in basedet, label id 80 means background, so minus 1 is needed
            classes = gt_boxes_per_img[:, 4].astype("int32") - 1
            boxes = Boxes(boxes)
            # affine transform might lead to empty boxes
            keep_idx = boxes.filter_by_size(sizes=0)

            boxes = boxes[keep_idx]
            classes = classes[keep_idx]
            num_boxes = boxes.shape[0]
            boxes.scale(self.box_scale)

            centers = boxes.centers
            centers_int = centers.astype("int32")
            gt_index[:num_boxes] = centers_int[..., 1] * self.output_size[1] + centers_int[..., 0]
            gt_reg[:num_boxes] = centers - centers_int
            reg_mask[:num_boxes] = 1

            wh = F.zeros_like(centers)
            wh[..., 0] = boxes.width
            wh[..., 1] = boxes.height
            CenterNetGT.generate_score_map(
                gt_scoremap, classes, wh, centers_int, self.min_overlap,
            )
            gt_wh[:num_boxes] = wh

            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        gt_dict = {
            "score_map": F.stack(scoremap_list, axis=0),
            "wh": F.stack(wh_list, axis=0),
            "reg": F.stack(reg_list, axis=0),
            "reg_mask": F.stack(reg_mask_list, axis=0),
            "index": F.stack(index_list, axis=0),
        }
        return gt_dict

    def get_losses(self, inputs):
        assert self.training
        inputs = self.pre_process(inputs)
        pred_dict = self.network_forward(inputs["image"])
        gt_dict = self.get_ground_truth(inputs)
        pred_score = pred_dict["cls"]
        loss_cls = modified_focal_loss(pred_score, gt_dict["score_map"])

        mask = gt_dict["reg_mask"]
        index = gt_dict["index"].astype("int32")

        # width and height loss, better version
        loss_wh = reg_l1_loss(pred_dict["wh"], mask, index, gt_dict["wh"])
        # regression loss
        loss_reg = reg_l1_loss(pred_dict["reg"], mask, index, gt_dict["reg"])

        loss_cls *= self.cfg.MODEL.LOSS.CLS_WEIGHT
        loss_wh *= self.cfg.MODEL.LOSS.WH_WEIGHT
        loss_reg *= self.cfg.MODEL.LOSS.REG_WEIGHT
        total_loss = loss_cls + loss_wh + loss_reg
        losses_dict = {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
        }
        return losses_dict

    def inference(self, inputs):
        assert not self.training
        inputs = self.pre_process(inputs)
        pred_dict = self.network_forward(inputs["image"])
        return self.post_process(pred_dict, inputs["img_info"])

    def post_process(self, pred_dict, img_info):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        fmap, reg, wh = pred_dict["cls"], pred_dict["reg"], pred_dict["wh"]

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).astype('int32')

        down_scale = self.cfg.MODEL.HEAD.DOWN_SCALE
        img_info = np.flip(img_info.numpy().flatten())  # (origin_w, origin_h, pad_w, pad_h)
        wh_center, wh_pad, wh_output = img_info[:2] // 2, img_info[-2:], img_info[-2:] // down_scale
        boxes = CenterNetDecoder.transform_boxes(boxes, wh_center, wh_pad, wh_output)

        container = Container(
            boxes=Boxes(mge.Tensor(boxes)),
            box_scores=scores,
            box_labels=classes,
        )

        return container


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.reshape(batch, channel, -1).transpose(0, 2, 1)

    num_channels = fmap.shape[-1]
    index = F.expand_dims(index, -1)
    index = F.repeat(index, repeats=num_channels, axis=index.ndim - 1)
    fmap = F.gather(fmap, 1, index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = F.expand_dims(mask, 2)
        mask = F.broadcast_to(mask, fmap.shape)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, num_channels)
    return fmap


def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)
    mask = F.expand_dims(mask, axis=2)
    mask = F.broadcast_to(mask, pred.shape).astype('float32')

    loss = F.loss.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (F.sum(mask) + 1e-4)
    return loss


def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = (gt == 1).astype("float32")
    neg_inds = (gt < 1).astype("float32")

    neg_weights = F.pow(1 - gt, 4)
    # clip min value is set to 1e-12 to maintain the numerical stability
    # clip max value is set to 1-1e-7 to maintain the numerical stability

    pred = F.clip(pred, lower=1e-12, upper=1 - 1e-7)
    pos_loss = F.log(pred) * F.pow(1 - pred, 2) * pos_inds
    neg_loss = F.log(1 - pred) * F.pow(pred, 2) * neg_weights * neg_inds
    num_pos = F.sum(pos_inds)
    pos_loss = F.sum(pos_loss)
    neg_loss = F.sum(neg_loss)

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


class CenterNetDecoder:

    @staticmethod
    def decode(fmap, wh, reg=None, cat_spec_wh=False, K=100):
        r"""
        decode output feature map to detection results
        Args:
            fmap(Tensor): output feature map
            wh(Tensor): tensor that represents predicted width-height
            reg(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `wh` or not
            K(int): topk value
        """
        batch, channel, height, width = fmap.shape

        fmap = CenterNetDecoder.pseudo_nms(fmap)

        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(fmap, K=K)
        if reg is not None:
            reg = gather_feature(reg, index, use_transform=True)
            reg = reg.reshape(batch, K, 2)
            xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.reshape(batch, K, 1) + 0.5
            ys = ys.reshape(batch, K, 1) + 0.5
        wh = gather_feature(wh, index, use_transform=True)

        if cat_spec_wh:
            wh = wh.reshape(batch, K, channel, 2)
            clses_ind = clses.reshape(batch, K, 1, 1).astype('int32')
            clses_ind = F.broadcast_to(clses_ind, (batch, K, 1, 2))
            wh = F.gather(wh, 2, clses_ind)
            wh = wh.reshape(batch, K, 2)
        else:
            wh = wh.reshape(batch, K, 2)

        clses = clses.reshape(batch, K, 1).astype('float32')
        scores = scores.reshape(batch, K, 1)

        half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2

        bboxes = F.concat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], axis=2)

        detections = (bboxes, scores, clses)

        return detections

    @staticmethod
    def transform_boxes(boxes, center, size, output_size, scale=1):
        r"""
        transform predicted boxes to target boxes
        Args:
            boxes(Tensor): torch Tensor with (Batch, N, 4) shape
            img_info(dict): dict contains all information of original image
            scale(float): used for multiscale testing
        """
        boxes = boxes.numpy().reshape(-1, 4)

        src, dst = CenterNetDecoder.generate_src_and_dst(center, size, output_size)

        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

        coords = boxes.reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        target_boxes = np.dot(aug_coords, trans.T).reshape(-1, 4)
        return target_boxes

    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms
        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        fmap = fmap.astype('float32')
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).astype('float32')
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = F.topk(scores.reshape(-1, height * width), K, descending=True)
        topk_scores = topk_scores.reshape(batch, channel, K)
        topk_inds = topk_inds.reshape(batch, channel, K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).astype('int32').astype('float32')
        topk_xs = (topk_inds % width).astype('int32').astype('float32')

        # get all topk in in a batch
        topk_scores, index = F.topk(topk_scores.reshape(batch, -1), K, descending=True)

        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).astype("int32")
        topk_inds = gather_feature(topk_inds.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


class CenterNetGT:

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetGT.get_gaussian_radius(gt_wh, min_overlap)
        radius = F.clip(radius, 0)
        radius = radius.astype("int32").numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            if channel_index >= 0:
                fmap[channel_index] = CenterNetGT.draw_gaussian(
                    fmap[channel_index], centers_int[i], radius[i]
                )
        return fmap

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a mge.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        if layers.is_empty_tensor(box_size):
            return mge.Tensor([])
        box_tensor = mge.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = F.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = F.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = F.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return F.min(F.stack([r1, r2, r3]), axis=0)

    @staticmethod
    def gaussian2D(radius, sigma=1):
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1

        gaussian = CenterNetGT.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = mge.Tensor(gaussian)

        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = F.maximum(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        return fmap
