#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os.path as osp
import megfile

from basedet.configs import FCOSConfig
from basedet.layers.backbone.basecls_adaptor import auto_convert_cfg_to_basecls

_suffix = osp.split(osp.realpath(__file__))[0].split("playground/")[-1]


class Cfg(FCOSConfig):

    def __init__(self):
        super().__init__()
        self.GLOBAL.OUTPUT_DIR = megfile.smart_path_join(
            "/data/Outputs/model_logs/basedet_playground", _suffix
        )
        self.GLOBAL.CKPT_SAVE_DIR = megfile.smart_path_join(self.GLOBAL.CKPT_SAVE_DIR, _suffix)
        auto_convert_cfg_to_basecls(self, "resnet18")
