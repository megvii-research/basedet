#!/usr/bin/env python3

import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.optimizer import AdamW

from basedet.utils import registers

from .default_solver import BaseSolver, Solver


@registers.solvers.register()
class DetrSolver(BaseSolver):

    @classmethod
    def build(cls, cfg, model):
        # build optimizer
        solver_cfg = cfg.SOLVER
        mode = solver_cfg.get("REDUCE_MODE")
        if mode is not None:
            cls.reduce_mode = mode
        assert cls.reduce_mode in ["MEAN", "SUM"]

        world_size = dist.get_world_size()
        lr = solver_cfg.BASIC_LR * cfg.MODEL.BATCHSIZE
        lr_backbone = solver_cfg.BACKBONE_LR * cfg.MODEL.BATCHSIZE
        wd = solver_cfg.WEIGHT_DECAY
        if cls.reduce_mode == "MEAN":
            lr *= world_size
            lr_backbone *= world_size
        else:
            wd *= world_size

        backbone_freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        backbone_params, non_backbone_params = [], []
        for name, param in model.named_parameters():
            if "backbone" in name:
                if "layer" not in name and backbone_freeze_at >= 1:
                    continue
                if "layer1" in name and backbone_freeze_at >= 2:
                    continue
                backbone_params.append(param)
            else:
                non_backbone_params.append(param)
        params_with_grad = backbone_params + non_backbone_params

        optimizer = AdamW(
            [
                {"params": non_backbone_params},
                {"params": backbone_params, "lr": lr_backbone},
            ],
            lr=lr,
            betas=solver_cfg.BETAS,
            weight_decay=wd,
        )
        # build grad_manager
        gm = GradManager()
        callbacks = (
            [dist.make_allreduce_cb(cls.reduce_mode, dist.WORLD)]
            if world_size > 1
            else None
        )  # noqa
        gm.attach(params_with_grad, callbacks=callbacks)

        scaler = cls.build_grad_scaler(cfg)
        return Solver(optimizer=optimizer, grad_manager=gm, grad_scaler=scaler)

    @classmethod
    def build_grad_scaler(cls, cfg):
        amp_cfg = cfg.TRAINER.AMP
        if amp_cfg.ENABLE:
            from megengine.amp import GradScaler
            scaler = (
                GradScaler(init_scale=65536.0, growth_interval=2000)
                if amp_cfg.DYNAMIC_SCALE
                else GradScaler(init_scale=128.0, growth_interval=0)
            )
            return scaler
        else:
            return None
