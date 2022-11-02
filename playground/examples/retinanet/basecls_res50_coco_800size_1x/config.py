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
        self.MODEL.BACKBONE.NAME = "basecls_resnet50"
        self.MODEL.BACKBONE.OUT_FEATURES = ["s2", "s3", "s4"]
        self.MODEL.FPN.TOP_BLOCK_IN_FEATURE = "s4"
        self.MODEL.WEIGHTS = "s3://basecls/zoo/resnet/resnet50/resnet50.pkl"
