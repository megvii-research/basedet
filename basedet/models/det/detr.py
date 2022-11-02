#!/usr/bin/env python3

import megengine as mge
import megengine.functional as F
import megengine.module as M

from basedet.layers import (
    MLP,
    HungarianMatcher,
    Transformer,
    build_backbone,
    build_pos_embed,
    iou_loss,
    linear_init,
    weighted_cross_entropy
)
from basedet.models import BaseNet
from basedet.structures import BoxConverter, Boxes, Container
from basedet.utils import all_reduce, registers


@registers.models.register()
class DETR(BaseNet):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg.MODEL
        self.backbone = build_backbone(model_cfg.BACKBONE)

        bockbone_channels = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
        }[model_cfg.BACKBONE.NAME]

        dim = model_cfg.TRANSFORMER.DIM
        self.pos_embed = build_pos_embed(pos_embed=model_cfg.POS_EMBED, N_steps=dim // 2)

        self.input_proj = M.Conv2d(bockbone_channels, dim, 1)
        self.query_embed = M.Embedding(model_cfg.NUM_QUERIES, dim)

        self.transformer = Transformer(
            dim=dim,
            num_heads=model_cfg.TRANSFORMER.NUM_HRADS,
            num_encoder_layers=model_cfg.TRANSFORMER.NUM_ENCODERS,
            num_decoder_layers=model_cfg.TRANSFORMER.NUM_DECODERS,
            dim_ffn=model_cfg.TRANSFORMER.DIM_FFN,
            dropout=model_cfg.TRANSFORMER.DROPOUT,
            normalize_before=model_cfg.TRANSFORMER.PRE_NORM,
            return_intermediate_dec=True,
        )

        self.class_embed = M.Linear(dim, cfg.DATA.NUM_CLASSES + 1)
        self.class_embed.apply(linear_init)
        self.bbox_embed = MLP(dim, dim, 4, 3)

        self.matcher = HungarianMatcher(
            model_cfg.MATCHER.SET_WEIGHT_CLASS,
            model_cfg.MATCHER.SET_WEIGHT_BBOX,
            model_cfg.MATCHER.SET_WEIGHT_GIOU,
        )

        self.category_weight = F.ones(cfg.DATA.NUM_CLASSES + 1)
        self.category_weight[-1] = cfg.LOSSES.EOS_COEF

        img_mean = self.cfg.MODEL.BACKBONE.IMG_MEAN
        if img_mean is not None:
            self.img_mean = mge.tensor(img_mean).reshape(1, -1, 1, 1)
        img_std = self.cfg.MODEL.BACKBONE.IMG_STD
        if img_std is not None:
            self.img_std = mge.tensor(img_std).reshape(1, -1, 1, 1)

    def pre_process(self, inputs):
        image = mge.Tensor(inputs["data"]) if isinstance(inputs, dict) else inputs
        if image.shape[1] == 4:
            image, mask = F.split(image, [3], axis=1)
        else:
            mask = F.zeros((len(image), 1, *image.shape[-2:]))
        image = super().pre_process(image)
        processed_data = {
            "image": image,
            "mask": mask,
            "img_info": mge.Tensor(inputs["im_info"]),
        }
        if self.training:
            processed_data.update(gt_boxes=mge.Tensor(inputs["gt_boxes"]))

        return processed_data

    def network_forward(self, image, mask=None):
        if mask is None:
            mask = F.zeros((len(image), 1, *image.shape[-2:]))

        src = self.backbone.extract_features(image)["res5"]
        mask = F.vision.interpolate(mask, size=src.shape[-2:], mode="nearest").astype(bool)
        mask = F.squeeze(mask, 1)
        pos = self.pos_embed(src, mask)

        hs, _ = self.transformer(
            self.input_proj(src),
            mask,
            self.query_embed.weight,
            pos,
        )

        outputs_class = self.class_embed(hs)
        outputs_coord = F.sigmoid(self.bbox_embed(hs))

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.cfg.LOSSES.AUX_LOSS:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return out

    def get_ground_truth(self, inputs):
        boxes, box_category = F.split(inputs["gt_boxes"], [4], 2)
        box_category = F.squeeze(box_category, -1).astype("int32") - 1
        WH = inputs["img_info"][:, [1, 0]]
        targets = []
        for bid in range(len(inputs["img_info"])):
            batch_boxes = boxes[bid].reshape(-1, 2, 2) / WH[bid]
            valid = batch_boxes[:, 0, :] < batch_boxes[:, 1, :]
            keep = valid[:, 0] & valid[:, 1]
            keep_boxes = batch_boxes.reshape(-1, 4)[keep]
            keep_boxes = BoxConverter.convert(keep_boxes, mode="xyxy2xcycwh")
            keep_category = box_category[bid][keep]
            tgt = {
                "boxes": keep_boxes,
                "boxes_category": keep_category,
            }
            targets.append(tgt)
        return targets

    def _get_src_permutation_idx(self, indices):
        batch_idx = F.concat(
            [F.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = F.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss_labels(self, outputs, targets, indices):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        target_classes_o = F.concat(
            [t["boxes_category"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = F.full(
            src_logits.shape[:2],
            self.cfg.DATA.NUM_CLASSES,
            dtype="int32",
        )
        idx = self._get_src_permutation_idx(indices)
        target_classes[idx] = target_classes_o
        loss_ce = weighted_cross_entropy(
            src_logits.transpose(0, 2, 1), target_classes, self.category_weight
        )
        return {"loss_ce": loss_ce}

    def get_loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = F.concat([t["boxes"][i] for t, (_, i) in zip(targets, indices)])
        if len(target_boxes):
            loss_bbox = F.loss.l1_loss(src_boxes, target_boxes, reduction="none")
            losses = {}
            losses["loss_bbox"] = loss_bbox.sum() / num_boxes
            _I = F.arange(len(src_boxes), dtype="int32")
            loss_giou = iou_loss(src_boxes, target_boxes, "xcycwh", "giou")[_I, _I]
            losses["loss_giou"] = loss_giou.sum() / num_boxes
            return losses
        return {"loss_bbox": mge.tensor(0.0), "loss_giou": mge.tensor(0.0)}

    def get_losses(self, inputs):
        assert self.training

        inputs = self.pre_process(inputs)
        outputs = self.network_forward(inputs["image"], inputs["mask"])
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        targets = self.get_ground_truth(inputs)
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = mge.tensor([num_boxes], dtype="float32")
        num_boxes = F.clip(all_reduce(num_boxes, mode="mean"), lower=1).detach()

        loss_dict = {}
        loss_dict.update(self.get_loss_labels(outputs, targets, indices))
        loss_dict.update(self.get_loss_boxes(outputs, targets, indices, num_boxes))

        coef_dict = {
            "loss_ce": self.cfg.LOSSES.CE_LOSS_COEF,
            "loss_bbox": self.cfg.LOSSES.BBOX_LOSS_COEF,
            "loss_giou": self.cfg.LOSSES.GIOU_LOSS_COEF,
        }

        if "aux_outputs" in outputs:
            aux_coef_dict = {}
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_loss_dict = {}
                aux_loss_dict.update(
                    self.get_loss_labels(aux_outputs, targets, aux_indices)
                )
                aux_loss_dict.update(
                    self.get_loss_boxes(aux_outputs, targets, aux_indices, num_boxes)
                )
                loss_dict.update({k + f"_{i}": v for k, v in aux_loss_dict.items()})
                aux_coef_dict.update({k + f"_{i}": v for k, v in coef_dict.items()})
            coef_dict.update(aux_coef_dict)

        loss_dict["total_loss"] = sum(
            [coef * loss_dict[k] for k, coef in coef_dict.items()]
        )

        return loss_dict

    def inference(self, inputs):
        assert not self.training

        inputs = self.pre_process(inputs)
        image = inputs["image"]
        assert image.shape[0] == 1
        outputs = self.network_forward(image, inputs["mask"])
        return self.post_process(outputs, inputs["img_info"])

    def post_process(self, outputs, img_info):
        out_logits, out_bbox = F.squeeze(outputs["pred_logits"]), F.squeeze(outputs["pred_boxes"])

        prob = F.softmax(out_logits, -1)
        scores = prob[:, :-1].max(-1)
        labels = F.argmax(prob[:, :-1], axis=-1)

        img_info = F.squeeze(img_info)
        img_h, img_w = img_info[2:4]
        boxes = BoxConverter.convert(out_bbox, mode="xcycwh2xyxy")
        scale_fct = F.stack([img_w, img_h, img_w, img_h])
        boxes = boxes * F.expand_dims(scale_fct, 0)
        return Container(
            boxes=Boxes(boxes).clip((img_h, img_w)),
            box_scores=scores,
            box_labels=labels,
        )
