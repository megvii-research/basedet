#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from ..detection_cfg import DetectionConfig

# This config file has many errors
TEST_SIZE = 416

_MODEL_CONFIG = dict(
    NAME="YOLOX",
    WEIGHTS="",
    DEPTH_FACTOR=1.0,
    WIDTH_FACTOR=1.0,
    DEPTHWISE=False,
    ACTIVATION="silu",
    BN_EPS=1e-3,
    BN_MOMENTUM=0.97,
    BACKBONE=dict(
        NAME="csp_darknet",
        OUT_FEATURES=["dark3", "dark4", "dark5"],
    ),
    FPN=dict(
        CONF_THRESHOLD=0.01,  # TEST
        NMS_THRESHOLD=0.5,
    ),
    # LOSSES=dict(),
    BATCHSIZE=8,
)


_YOLOX_CONFIG = dict(
    MODEL=_MODEL_CONFIG,
    SOLVER=dict(
        # TODO wangfeng: check
        BUILDER_NAME="YOLOXSolver",
        REDUCE_MODE="MEAN",
        BASIC_LR=0.01 / (_MODEL_CONFIG["BATCHSIZE"] * 8),  # The basic lr for single-image
        MIN_LR_RATIO=0.05,
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0005,
        WARM_EPOCH=5,
        NUM_IMAGE_PER_EPOCH=120000,
        MAX_EPOCH=300,
        LR_DECAY_RATE=0.1,
    ),
    DATA=dict(
        BUILDER_NAME="YOLOXDataloaderBuilder",
        TRAIN=dict(
            name="coco_2017_train",
            remove_images_without_annotations=True,
            order=("image", "boxes", "boxes_category", "info"),
        ),
        TEST=dict(
            name="coco_2017_val",
            remove_images_without_annotations=False,
            order=("image", "info"),
        ),
        NUM_CLASSES=80,
        NUM_WORKERS=4,
        ENABLE_INFINITE_SAMPLER=True,
    ),
    TRAINER=dict(
        EMA=dict(
            ENABLE=True,
        ),
    ),
    HOOKS=dict(
        BUILDER_NAME="YOLOXHookList",
    ),
    AUG=dict(
        TRAIN_VALUE=(
            # ("ResizeAndPad", dict(target_size=(416, 416), pad_value=114)),
        ),
        TRAIN_SETTING=dict(
            INPUT_SIZE=(640, 640),
            MULTISCALE_RANGE=(14, 26),  # multiply 32 to final range
            SYNC_ITER=10,
            MOSAIC_PROB=1.0,
            MOSAIC_SCALE=(0.1, 2),
            ENABLE_MIXUP=True,
            MIXUP_PROB=1.0,
            MIXUP_SCALE=(0.5, 1.5),
            HSV_PROB=1.0,
            FLIP_PROB=0.5,
            DEGREES=10.0,
            TRANSLATE=0.1,
            SHEAR=2.0,
            NO_AUG_EPOCH=15,
        ),
        TRAIN_WRAPPER=(
            ("MGE_Compose", dict(order=("image", "boxes", "boxes_category"))),
        ),
    ),
    TEST=dict(
        CLS_THRESHOLD=0.001,
        IMG_MIN_SIZE=TEST_SIZE,
        IMG_MAX_SIZE=TEST_SIZE,
    ),
)


class YOLOXConfig(DetectionConfig):

    def __init__(self):
        super().__init__()
        del self.MODEL
        self.merge(_YOLOX_CONFIG)
        self.GLOBAL.LOG_INTERVAL = 10
        del self.SOLVER.WARM_ITERS

        self.TEST.AUG.VALUE = (
            (
                "MGE_ShortestEdgeResize",
                dict(min_size=TEST_SIZE, max_size=TEST_SIZE, sample_style="choice"),
            ),
            ("PadToTargetSize", dict(target_size=(TEST_SIZE, TEST_SIZE), pad_value=114.0)),
            ("ToMode", dict(mode="NCHW")),
        )
