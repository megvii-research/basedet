#!/usr/bin/python3

import megengine.functional as F

from basedet.structures import Boxes
from basedet.utils import registers

from .fcos import FCOS


@registers.models.register()
class ATSS(FCOS):
    """
    Implement ATSS (https://arxiv.org/abs/1912.02424).
    """

    def get_ground_truth(self, anchors_list, batched_gt_boxes, batched_num_gts):
        labels_list = []
        offsets_list = []
        ctrness_list = []

        all_level_anchors = F.concat(anchors_list, axis=0)
        for boxes_with_labels, num_boxes in zip(batched_gt_boxes, batched_num_gts):
            gt_boxes_with_labels = boxes_with_labels[:num_boxes]
            gt_boxes = Boxes(gt_boxes_with_labels[:, :4])

            ious = []
            candidate_idxs = []
            base = 0
            for stride, anchors_i in zip(self.head.strides, anchors_list):
                ious.append(
                    gt_boxes.iou(
                        F.concat([
                            anchors_i - stride * self.cfg.MODEL.ANCHOR.SCALE / 2,
                            anchors_i + stride * self.cfg.MODEL.ANCHOR.SCALE / 2,
                        ], axis=1)
                    )
                )
                gt_centers = gt_boxes.centers
                distances = F.sqrt(
                    F.sum((F.expand_dims(gt_centers, axis=1) - anchors_i) ** 2, axis=2)
                )
                _, topk_idxs = F.topk(distances, self.cfg.MODEL.ANCHOR.TOPK)
                candidate_idxs.append(base + topk_idxs)
                base += anchors_i.shape[0]
            ious = F.concat(ious, axis=1)
            candidate_idxs = F.concat(candidate_idxs, axis=1)

            candidate_ious = F.gather(ious, 1, candidate_idxs)
            ious_thr = (F.mean(candidate_ious, axis=1, keepdims=True)
                        + F.std(candidate_ious, axis=1, keepdims=True))
            is_foreground = F.scatter(
                F.zeros(ious.shape), 1, candidate_idxs, F.ones(candidate_idxs.shape)
            ).astype(bool) & (ious >= ious_thr)

            is_in_boxes = F.min(self.box_coder.encode(
                all_level_anchors, F.expand_dims(gt_boxes, axis=1)
            ), axis=2) > 0

            ious[~is_foreground] = -1
            ious[~is_in_boxes] = -1

            match_indices = F.argmax(ious, axis=0)
            gt_boxes_matched = gt_boxes_with_labels[match_indices]
            anchor_max_iou = F.indexing_one_hot(ious, match_indices, axis=0)

            labels = gt_boxes_matched[:, 4].astype("int32")
            labels[anchor_max_iou == -1] = 0
            offsets = self.box_coder.encode(all_level_anchors, gt_boxes_matched[:, :4])

            left_right = offsets[:, [0, 2]]
            top_bottom = offsets[:, [1, 3]]
            ctrness = F.sqrt(
                F.clip(F.min(left_right, axis=1) / F.max(left_right, axis=1), lower=0)
                * F.clip(F.min(top_bottom, axis=1) / F.max(top_bottom, axis=1), lower=0)
            )

            labels_list.append(labels)
            offsets_list.append(offsets)
            ctrness_list.append(ctrness)

        return (
            F.stack(labels_list, axis=0).detach(),
            F.stack(offsets_list, axis=0).detach(),
            F.stack(ctrness_list, axis=0).detach(),
        )
