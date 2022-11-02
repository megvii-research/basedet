#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import megfile

from basedet.configs import RetinaNetConfig

_suffix = os.path.split(os.path.realpath(__file__))[0].split("playground/")[-1]


class Cfg(RetinaNetConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GLOBAL.OUTPUT_DIR = megfile.smart_path_join(
            "/data/Outputs/model_logs/basedet_playground", _suffix
        )
        self.GLOBAL.CKPT_SAVE_DIR = megfile.smart_path_join(self.GLOBAL.CKPT_SAVE_DIR, _suffix)
