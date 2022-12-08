#!/usr/bin/env python3

from ..detection_cfg import DetectionConfig

# This config file has many errors
_MODEL_CONFIG = dict(
    NAME="YOLOv3",
    WEIGHTS="s3://basedet/backbone/darknet53.pkl",
    BACKBONE=dict(
        NAME="darknet53",
        OUT_FEATURES=["dark3", "dark4", "dark5"],
        IMG_MEAN=(0.485, 0.456, 0.406),
        IMG_STD=(0.229, 0.224, 0.225),
        NORM="BN",
        FREEZE_AT=0,
    ),
    FPN=dict(
        CONF_THRESHOLD=0.01,  # TEST
        NMS_THRESHOLD=0.5,
    ),
    ANCHOR=dict(
        SCALES=[
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [42, 119]],
            [[10, 13], [16, 30], [33, 23]],
        ],
    ),
    LOSSES=dict(),
    NMS_TYPE='normal',
    BATCHSIZE=8,
    IGNORE_THRESHOLD=0.7,
)


_YOLOv3_CONFIG = dict(
    MODEL=_MODEL_CONFIG,
    SOLVER=dict(
        # TODO wangfeng: check
        BUILDER_NAME="DefaultSolver",
        REDUCE_MODE="MEAN",
        BASIC_LR=0.001 / _MODEL_CONFIG["BATCHSIZE"],  # The basic learning rate for single-image
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0005,
        WARM_ITERS=2000,
        NUM_IMAGE_PER_EPOCH=100000,
        MAX_EPOCH=320,
        LR_DECAY_STAGES=[256, 300],
        LR_DECAY_RATE=0.1,
    ),
    DATA=dict(
        BUILDER_NAME="DataloaderBuilder",
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
        NUM_WORKERS=2,
        ENABLE_INFINITE_SAMPLER=True,
    ),
    AUG=dict(
        TRAIN_VALUE=(
            ("RandomBrightness", dict(value=32.0 / 255, prob=0.5)),
            ("RandomContrast", dict(value=0.5, prob=0.5,)),
            ("RandomSaturation", dict(value=0.5, prob=0.5,)),
            ("MinIoURandomCrop", dict(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            ("Expand", dict(ratio_range=(2, 4), mean=[123.675, 116.280, 103.530], prob=0.6)),
            ("MGE_Resize", dict(output_size=(512, 512))),
            ("MGE_RandomHorizontalFlip", dict(prob=0.5)),
            ("MGE_ToMode", dict(mode="CHW")),
        ),
        TRAIN_WRAPPER=(("MGE_Compose", dict(order=("image", "boxes", "boxes_category"),)),),
        TEST_VALUE=(
            ('MGE_Resize', dict(output_size=(608, 608))),
            ("MGE_ToMode", dict(mode="CHW")),
        ),
    ),
)


class YOLOv3Config(DetectionConfig):

    def __init__(self):
        super().__init__()
        self.merge(_YOLOv3_CONFIG)
