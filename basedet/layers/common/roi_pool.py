#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
from typing import List, Tuple

import megengine.functional as F
from megengine import Tensor

__all__ = ["roi_pool"]


def assign_rois(rois, strides) -> Tuple[Tensor]:
    rois = rois.detach()
    canonical_level = 4
    canonical_box_size = 224
    min_level, max_level = int(math.log2(strides[0])), int(math.log2(strides[-1]))

    num_fms = len(strides)
    box_area = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
    assigned_level = F.floor(
        canonical_level + F.log(F.sqrt(box_area) / canonical_box_size) / math.log(2)
    ).astype("int32")
    assigned_level = F.minimum(assigned_level, max_level)
    assigned_level = F.maximum(assigned_level, min_level)
    assigned_level = assigned_level - min_level

    # avoid empty assignment
    assigned_level = F.concat(
        [assigned_level, F.arange(num_fms, dtype="int32", device=assigned_level.device)],
    )
    rois = F.concat([rois, F.zeros((num_fms, rois.shape[-1]))])
    return rois, assigned_level


def roi_pool(
    features: List[Tensor],
    rois: Tensor,
    strides: List[int],
    pool_shape: List[int],
    pooler_type: str = "roi_align",
) -> Tensor:
    """
    RoI Pooling logic. Include roi assign and pooling.

    Args:
        features: input features.
        rois: shape (N, 5). First column is the box index. The next 4 columns are ``xyxy``.
        strides: stride of input features.
        pool_shape: output shape of RoI feature.
        pooler_type: type of RoI pooler. Only support "roi_align" and "roi_pool" now.
            Defaults to "roi_align".
    """
    assert pooler_type in ("roi_align", "roi_pool")
    assert len(strides) == len(features)

    rois, assigned_level = assign_rois(rois.detach(), strides)

    pool_list, inds_list = [], []
    for i, (feat, stride) in enumerate(zip(features, strides)):
        _, inds = F.cond_take(assigned_level == i, assigned_level)
        level_rois = rois[inds]

        scale = 1.0 / stride
        if pooler_type == "roi_pool":
            pool_fm = F.nn.roi_pooling(feat, level_rois, pool_shape, mode="max", scale=scale)
        elif pooler_type == "roi_align":
            pool_fm = F.nn.roi_align(
                feat, level_rois, pool_shape, mode="average",
                spatial_scale=scale, sample_points=2, aligned=True,
            )
        pool_list.append(pool_fm)
        inds_list.append(inds)

    fm_order = F.argsort(F.concat(inds_list, axis=0))
    pool_feature = F.concat(pool_list, axis=0)
    pool_feature = pool_feature[fm_order][:-len(features)]

    return pool_feature
