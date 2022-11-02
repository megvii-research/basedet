#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ..detection_cfg import DetectionConfig

_CENTERNET_CONFIG = dict(
    MODEL=dict(
        NAME="CenterNet",
        WEIGHTS="s3://basedet/backbone/resnet50_fbaug_633cb650.pkl",
        BATCHSIZE=16,
        BACKBONE=dict(
            NAME="resnet50",
            IMG_MEAN=[103.53, 116.28, 123.675],
            IMG_STD=[57.375, 57.120, 58.395],
            NORM="BN",
            FREEZE_AT=0,
        ),
        HEAD=dict(
            DECONV_CHANNEL=[2048, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            MODULATE_DEFORM=True,
            IN_CHANNELS=64,
            CLS_PRIOR_PROB=0.1,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=128,
        ),
        LOSS=dict(
            CLS_WEIGHT=1,
            WH_WEIGHT=0.1,
            REG_WEIGHT=1,
        ),
        OUTPUT_SIZE=(128, 128),
    ),
    AUG=dict(
        TRAIN_VALUE=(
            ("CenterAffine", dict(border=128, output_size=(512, 512))),
            ("MGE_RandomHorizontalFlip", dict(prob=0.5)),
            ("MGE_BrightnessTransform", dict(value=0.4)),
            ("MGE_ContrastTransform", dict(value=0.4)),
            ("MGE_SaturationTransform", dict(value=0.4)),
            ("MGE_Lighting", dict(scale=0.1)),
            ("MGE_ToMode", dict(mode="CHW")),
        ),
    ),
    DATA=dict(
        NUM_WORKERS=4,
    ),
    SOLVER=dict(
        MAX_EPOCH=140,
        BASIC_LR=0.02 / (8 * 16),
        WEIGHT_DECAY=1e-4,
        LR_DECAY_RATE=0.1,
        LR_DECAY_STAGES=[90, 120],
        WARM_ITERS=1000,
        NUM_IMAGE_PER_EPOCH=115200,
    ),
)


class CenterNetConfig(DetectionConfig):
    def __init__(self):
        super().__init__()
        self.TEST.AUG.VALUE = (
            ("TestTimeCenterPad", dict()),
            ("ToMode", dict(mode="NCHW")),
        )
        self.merge(_CENTERNET_CONFIG)
