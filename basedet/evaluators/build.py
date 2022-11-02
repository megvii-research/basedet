#!/usr/bin/env python3
from .coco_eval import COCOEvaluator

__all__ = ["build_evaluator"]


def build_evaluator(cfg):
    """
    Build evaluator to evaluate inference result.

    Args:
        cfg (BaseConfig): config of model.
    """
    return COCOEvaluator(cfg)
