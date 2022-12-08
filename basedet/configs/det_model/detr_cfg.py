#!/usr/bin/env python3

import megengine.data.transform as T

from basedet.data.transforms import RandomSizeCrop

from ..detection_cfg import DetectionConfig

_DETR_CONFIG = dict(
    MODEL=dict(
        NAME="DETR",
        WEIGHTS="s3://basedet/backbone/resnet50_fbaug_633cb650.pkl",
        POS_EMBED="sine",
        NUM_QUERIES=100,
        TRANSFORMER=dict(
            DIM=256,
            NUM_HRADS=8,
            NUM_ENCODERS=6,
            NUM_DECODERS=6,
            DIM_FFN=2048,
            DROPOUT=0.1,
            PRE_NORM=False,
        ),
        MATCHER=dict(
            SET_WEIGHT_CLASS=1,
            SET_WEIGHT_BBOX=5,
            SET_WEIGHT_GIOU=2,
        ),
    ),
    LOSSES=dict(
        AUX_LOSS=True,
        CE_LOSS_COEF=1,
        BBOX_LOSS_COEF=5,
        GIOU_LOSS_COEF=2,
        EOS_COEF=0.1,
    ),
    DATA=dict(
        BUILDER_NAME="DETRDataloaderBuilder",
    ),
    AUG=dict(
        TRAIN_VALUE=(
            ("MGE_RandomHorizontalFlip", dict(prob=0.5)),
            (
                "RandomSelect",
                dict(
                    transforms=[
                        T.ShortestEdgeResize(
                            min_size=(
                                480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800,
                            ),
                            max_size=1333,
                            sample_style="choice",
                        ),
                        T.Compose(
                            [
                                T.ShortestEdgeResize(
                                    min_size=(400, 500, 600),
                                    max_size=float("inf"),
                                    sample_style="choice",
                                ),
                                RandomSizeCrop(384, 600),
                                T.ShortestEdgeResize(
                                    min_size=(
                                        480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800,
                                    ),
                                    max_size=1333,
                                    sample_style="choice",
                                ),
                            ]
                        ),
                    ]
                ),
            ),
            ("MGE_ToMode", dict(mode="CHW")),
        ),
    ),
    SOLVER=dict(
        BUILDER_NAME="DetrSolver",
        REDUCE_MODE="MEAN",
        BASIC_LR=1e-4 / 16.0,
        BACKBONE_LR=1e-5 / 16.0,
        BETAS=(0.9, 0.999),
        WEIGHT_DECAY=1e-4,
        WARM_ITERS=0,
        NUM_IMAGE_PER_EPOCH=118287,
        MAX_EPOCH=150,
        LR_DECAY_STAGES=[100],
        LR_DECAY_RATE=0.1,
    ),
    TRAINER=dict(
        GRAD_CLIP=dict(
            ENABLE=True,
            TYPE="norm",
            # ARGS=dict(max_norm=0.1, ord=2),
        )
    ),
)


class DETRConfig(DetectionConfig):
    def __init__(self):
        super().__init__()
        self.merge(_DETR_CONFIG)
        self.TRAINER.GRAD_CLIP.ARGS = dict(max_norm=0.1, ord=2)
