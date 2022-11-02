#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine as mge
import megengine.distributed as dist
import megengine.module as M
from megengine.optimizer import SGD

from basedet.utils import registers

from .default_solver import DefaultSolver


@registers.solvers.register()
class YOLOXSolver(DefaultSolver):

    @classmethod
    def build_optimizer(cls, cfg, model):
        solver_cfg = cfg.SOLVER
        lr = solver_cfg.BASIC_LR * cfg.MODEL.BATCHSIZE * dist.get_world_size()
        wd = solver_cfg.WEIGHT_DECAY

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, mge.Parameter):
                pg2.append(v.bias)  # biases

            if isinstance(v, M.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, mge.Parameter):
                pg1.append(v.weight)  # apply decay

        pg_list = []
        pg_list.append({"params": pg0 + pg2})
        pg_list.append({"params": pg1, "weight_decay": wd})
        optimizer = SGD(pg_list, lr=float(lr), momentum=solver_cfg.MOMENTUM, nesterov=True)
        return optimizer
