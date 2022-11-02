#!/usr/bin/env python3

from basecore.config import ConfigDict

from .base_cfg import BaseConfig
from .det_model import *
from .detection_cfg import DetectionConfig

__all__ = [k for k in globals().keys() if not k.startswith("_")]
