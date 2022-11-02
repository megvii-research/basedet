#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import megfile

from basedet.configs import RetinaNetConfig

_suffix = os.path.split(os.path.realpath(__file__))[0].split("playground/")[-1]

_OBJ365_DELTA = dict(
    MODEL=dict(
        # BACKBONE=dict(NORM="SyncBN"),
        # FPN=dict(NORM="SyncBN"),
    ),
    DATA=dict(
        TRAIN=dict(
            name="objects365_train",
            remove_images_without_annotations=True,
            order=("image", "boxes", "boxes_category", "info"),
        ),
        TEST=dict(
            name="objects365_val",
            remove_images_without_annotations=False,
            order=("image", "info"),
        ),
        NUM_CLASSES=365,
    ),
    SOLVER=dict(
        NUM_IMAGE_PER_EPOCH=600464,
        MAX_EPOCH=24,
        LR_DECAY_STAGES=(16, 22),
    ),
    GLOBAL=dict(
        OUTPUT_DIR=megfile.smart_path_join(
            "/data/Outputs/model_logs/basedet_playground", _suffix
        ),
    ),
)


class Cfg(RetinaNetConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.merge(_OBJ365_DELTA)
        self.GLOBAL.CKPT_SAVE_DIR = megfile.smart_path_join(self.GLOBAL.CKPT_SAVE_DIR, _suffix)
