#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os.path as osp
import megfile

from basedet.configs import YOLOXConfig

_suffix = osp.split(osp.realpath(__file__))[0].split("playground/")[-1]

_tiny_value = dict(
    MODEL=dict(
        DEPTH_FACTOR=0.33,
        WIDTH_FACTOR=0.375,
    ),
)


class Cfg(YOLOXConfig):

    def __init__(self):
        super().__init__()
        self.GLOBAL.OUTPUT_DIR = megfile.smart_path_join(
            "/data/Outputs/model_logs/basedet_playground", _suffix
        )
        self.GLOBAL.CKPT_SAVE_DIR = megfile.smart_path_join(self.GLOBAL.CKPT_SAVE_DIR, _suffix)
        self.merge(_tiny_value)
        self.AUG.TRAIN_SETTING.ENABLE_MIXUP = False
