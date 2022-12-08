#!/usr/bin/env python3

from .fcos_cfg import FCOSConfig


class OTAConfig(FCOSConfig):

    def __init__(self):
        super().__init__()
        self.MODEL.NAME = "OTA"
        self.MODEL.HEAD.WITH_NORM = True
        self.MODEL.HEAD.SHARE_PARAM = True
        self.MODEL.HEAD.NORM_REG_TARGETS = True
        self.MODEL.MATCHING = "topk"  # could be "topk", "sinkhorn"
