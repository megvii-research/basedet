#!/usr/bin/env python3

from .fcos_cfg import FCOSConfig

_ATSS_CONFIG = dict(
    MODEL=dict(
        NAME="ATSS",
        ANCHOR=dict(
            SCALE=8,
            TOPK=9,
        ),
        LOSSES=dict(
            REG_LOSS_WEIGHT=2.0,
        ),
    ),
)


class ATSSConfig(FCOSConfig):

    def __init__(self):
        super().__init__()
        self.merge(_ATSS_CONFIG)
        del self.MODEL.OBJECT_SIZES_OF_INTEREST
        del self.MODEL.CENTER_SAMPLING_RADIUS
