#!/usr/bin/python3
# -*- coding:utf-8 -*-

from .retinanet_cfg import RetinaNetConfig

_FREEANCHOR_CONFIG = dict(
    MODEL=dict(
        NAME="FreeAnchor",
        WEIGHTS="s3://basedet/backbone/resnet50_fbaug_633cb650.pkl",
        LOSSES=dict(
            FOCLA_LOSS_ALPHA=0.5,
            FOCAL_LOSS_GAMMA=2,
            SMOOTH_L1_BETA=0.0,
            REG_LOSS_WEIGHT=0.75,
        ),
        BOX_REG=dict(
            STD=[0.1, 0.1, 0.2, 0.2],
        ),
        HEAD=dict(
            CLS_PRIOR_PROB=0.02,
        ),
        BUCKET=dict(
            BOX_IOU_THRESH=0.6,
            BUCKET_SIZE=50,
        ),
    ),
)


class FreeAnchorConfig(RetinaNetConfig):

    def __init__(self):
        super().__init__()
        self.merge(_FREEANCHOR_CONFIG)
