#!/usr/bin/env python3

from ..detection_cfg import DetectionConfig

_RETINANET_CONFIG = dict(
    MODEL=dict(
        NAME="RetinaNet",
        WEIGHTS="s3://basedet/backbone/resnet50_fbaug_633cb650.pkl",
        BACKBONE=dict(
            OUT_FEATURES=["res3", "res4", "res5"],
            OUT_FEATURE_CHANNELS=[512, 1024, 2048],
        ),
        FPN=dict(
            OUT_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NORM=None,
            STRIDES=[8, 16, 32, 64, 128],
            TOP_BLOCK_IN_CHANNELS=2048,
            TOP_BLOCK_IN_FEATURE="res5",
            OUT_CHANNELS=256,
        ),
        ANCHOR=dict(
            SCALES=[
                [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
                for x in [32, 64, 128, 256, 512]
            ],
            RATIOS=[[0.5, 1, 2]],
            OFFSET=0.5,
        ),
        LOSSES=dict(
            FOCAL_LOSS_ALPHA=0.25,
            FOCAL_LOSS_GAMMA=2,
            SMOOTH_L1_BETA=0.0,  # use L1 loss
            REG_LOSS_WEIGHT=1.0,
        ),
        BOX_REG=dict(
            MEAN=[0.0, 0.0, 0.0, 0.0],
            STD=[1.0, 1.0, 1.0, 1.0],
        ),
        MATCHER=dict(
            THRESHOLDS=[0.4, 0.5],
            LABELS=[0, -1, 1],
            ALLOW_LOW_QUALITY=True,
        ),
        HEAD=dict(
            NUM_CONVS=4,
            CLS_PRIOR_PROB=0.01,
        ),
    ),
)


class RetinaNetConfig(DetectionConfig):

    def __init__(self):
        super().__init__()
        self.merge(_RETINANET_CONFIG)
