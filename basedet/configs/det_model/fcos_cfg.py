#!/usr/bin/python3
# -*- coding:utf-8 -*-

from ..detection_cfg import DetectionConfig
from ..extra_cfg import ModelConfig


class FCOSModel(ModelConfig):

    def __init__(self):
        super().__init__()
        self.NAME = "FCOS"
        self.WEIGHTS = "s3://basedet/backbone/resnet50_fbaug_633cb650.pkl"
        self.ANCHOR = dict(
            NUM_ANCHORS=1,
            OFFSET=0.5,
        )
        self.BACKBONE.update(
            dict(
                OUT_FEATURES=["res3", "res4", "res5"],
                OUT_FEATURE_CHANNELS=[512, 1024, 2048],
            )
        )
        self.FPN = dict(
            OUT_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NORM=None,
            STRIDES=[8, 16, 32, 64, 128],
            TOP_BLOCK_IN_CHANNELS=2048,
            OUT_CHANNELS=256,
            TOP_BLOCK_IN_FEATURE="res5",
        )
        self.LOSSES = dict(
            FOCAL_LOSS_ALPHA=0.25,
            FOCAL_LOSS_GAMMA=2,
            IOU_LOSS_TYPE="giou",
            REG_LOSS_WEIGHT=1.0,
        )
        self.BOX_REG = dict(
            MEAN=[0.0, 0.0, 0.0, 0.0],
            STD=[1.0, 1.0, 1.0, 1.0],  # check box reg
        )
        self.HEAD = dict(
            NUM_CONVS=4,
            CLS_PRIOR_PROB=0.01,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64], [64, 128], [128, 256], [256, 512], [512, float("inf")]
            ],
            CENTER_SAMPLING_RADIUS=1.5,
        )


class FCOSConfig(DetectionConfig):

    def __init__(self):
        super().__init__()
        self.MODEL: FCOSModel = FCOSModel()
        self.TEST.IOU_THRESHOLD = 0.6
