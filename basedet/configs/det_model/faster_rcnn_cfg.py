#!/usr/bin/env python3

from ..detection_cfg import DetectionConfig

_FASTER_RCNN_CONFIG = dict(
    MODEL=dict(
        NAME="FasterRCNN",
        WEIGHTS="s3://basedet/backbone/resnet50_fbaug_633cb650.pkl",
        BACKBONE=dict(
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
        ),
        FPN=dict(
            OUT_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            NORM=None,
            STRIDES=[4, 8, 16, 32, 64],
            TOP_BLOCK_IN_CHANNELS=2048,
            OUT_CHANNELS=256,
            TOP_BLOCK_IN_FEATURE="p5",
        ),
        RPN=dict(
            CHANNELS=256,
            NMS_THRESHOLD=0.7,
            NUM_SAMPLE_ANCHORS=256,
            POSITIVE_ANCHOR_RATIO=0.5,
            TRAIN_PREV_NMS_TOPK=2000,
            TRAIN_POST_NMS_TOPK=1000,
            TEST_PREV_NMS_TOPK=1000,
            TEST_POST_NMS_TOPK=1000,
        ),
        ROI_POOLER=dict(
            METHOD="roi_align",
            SIZE=(7, 7),
        ),
        RCNN=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            STRIDES=[4, 8, 16, 32],
            NUM_ROIS=512,
            FG_RATIO=0.5,
            FG_THRESHOLD=0.5,
            BG_THRESHOLD_HIGH=0.5,
            BG_THRESHOLD_LOW=0.0,
        ),
        ANCHOR=dict(
            SCALES=[[x] for x in [32, 64, 128, 256, 512]],
            RATIOS=[[0.5, 1, 2]],
            OFFSET=0.5,
        ),
        LOSSES=dict(
            RPN_SMOOTH_L1_BETA=0,  # use L1 loss
            RCNN_SMOOTH_L1_BETA=0,  # use L1 loss
        ),
        RPN_BOX_REG=dict(
            MEAN=[0.0, 0.0, 0.0, 0.0],
            STD=[1.0, 1.0, 1.0, 1.0],
        ),
        RCNN_BOX_REG=dict(
            MEAN=[0.0, 0.0, 0.0, 0.0],
            STD=[0.1, 0.1, 0.2, 0.2],
        ),
        MATCHER=dict(
            THRESHOLDS=[0.3, 0.7],
            LABELS=[0, -1, 1],
            ALLOW_LOW_QUALITY=True,
        ),
    ),
    SOLVER=dict(
        BUILDER_NAME="DetSolver",
        REDUCE_MODE="MEAN",
        BASIC_LR=0.02 / 16,  # The basic learning rate for single-image
        MOMENTUM=0.9,
        WEIGHT_DECAY=1e-4,
        WARM_ITERS=500,
        NUM_IMAGE_PER_EPOCH=80000,
        MAX_EPOCH=18,
        LR_DECAY_STAGES=[12, 16],
        LR_DECAY_RATE=0.1,
    ),
)


class FasterRCNNConfig(DetectionConfig):

    def __init__(self):
        super().__init__()
        self.merge(_FASTER_RCNN_CONFIG)
