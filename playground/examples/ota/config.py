#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os.path as osp
import megfile

from basedet.configs import OTAConfig

_suffix = osp.split(osp.realpath(__file__))[0].split("playground/")[-1]


class Cfg(OTAConfig):

    def __init__(self):
        super().__init__()
        self.GLOBAL.OUTPUT_DIR = megfile.smart_path_join(
            "/data/Outputs/model_logs/basedet_playground", _suffix
        )
        self.GLOBAL.CKPT_SAVE_DIR = megfile.smart_path_join(self.GLOBAL.CKPT_SAVE_DIR, _suffix)
