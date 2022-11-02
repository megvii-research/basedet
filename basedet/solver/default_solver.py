#!/usr/bin/env python3
import megengine.distributed as dist
import megengine.module as M
import megengine.optimizer as optim
from megengine.autodiff import GradManager

from basecore.engine import Solver

from basedet.configs import DetectionConfig
from basedet.utils import registers

__all__ = ["Solver", "BaseSolver", "DefaultSolver", "DetSolver"]


class BaseSolver:

    @classmethod
    def build(cls, cfg, model):
        raise NotImplementedError


@registers.solvers.register()
class DefaultSolver(BaseSolver):

    @classmethod
    def build(cls, cfg: DetectionConfig, model: M.Module):
        """build default solver by provided model and configuration."""
        solver_cfg = cfg.SOLVER
        cls.reduce_mode = solver_cfg.get("REDUCE_MODE", "MEAN")
        assert cls.reduce_mode in ["MEAN", "SUM"]

        optimizer = cls.build_optimizer(cfg, model)
        gm = cls.build_grad_manager(cfg, model)
        scaler = cls.build_grad_scaler(cfg.TRAINER.AMP)
        return Solver(optimizer, grad_manager=gm, grad_scaler=scaler)

    @classmethod
    def build_optimizer(cls, cfg, model: M.Module):
        """build optimizer to optimize model parameters"""
        lr = cfg.SOLVER.BASIC_LR * cfg.MODEL.BATCHSIZE
        wd = cfg.SOLVER.WEIGHT_DECAY
        world_size = dist.get_world_size()
        if cls.reduce_mode == "MEAN":
            lr = lr * world_size
        else:
            wd = wd * world_size
        optimizer_name = cfg.SOLVER.get("OPTIMIZER_NAME", "SGD")
        extra_args = cfg.SOLVER.get("EXTRA_OPT_ARGS", dict())
        optimizer = getattr(optim, optimizer_name)(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            **extra_args,
        )
        return optimizer

    @classmethod
    def build_grad_manager(cls, cfg, model: M.Module):
        gm = GradManager()
        world_size = dist.get_world_size()
        callbacks = [dist.make_allreduce_cb(cls.reduce_mode, dist.WORLD)] if world_size > 1 else None  # noqa
        gm.attach(model.parameters(), callbacks=callbacks)
        return gm

    @classmethod
    def build_grad_scaler(cls, amp_cfg):
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


@registers.solvers.register()
class DetSolver(DefaultSolver):

    @classmethod
    def params(cls, cfg, model: M.Module):
        """get all parameters of model to update"""
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        params_with_grad = []
        for name, param in model.named_parameters():
            if "bottom_up.conv1" in name and freeze_at >= 1:
                continue
            if "bottom_up.layer1" in name and freeze_at >= 2:
                continue
            params_with_grad.append(param)

        return params_with_grad

    @classmethod
    def build_optimizer(cls, cfg, model: M.Module):
        """build optimizer to optimize model parameters"""

        lr = cfg.SOLVER.BASIC_LR * cfg.MODEL.BATCHSIZE
        wd = cfg.SOLVER.WEIGHT_DECAY
        world_size = dist.get_world_size()
        if cls.reduce_mode == "MEAN":
            lr = lr * world_size
        else:
            wd = wd * world_size
        optimizer_name = cfg.SOLVER.get("OPTIMIZER_NAME", "SGD")
        extra_args = cfg.SOLVER.get("EXTRA_OPT_ARGS", dict())
        optimizer = getattr(optim, optimizer_name)(
            cls.params(cfg, model),
            lr=lr,
            weight_decay=wd,
            **extra_args,
        )
        return optimizer

    @classmethod
    def build_grad_manager(cls, cfg, model: M.Module):
        gm = GradManager()
        world_size = dist.get_world_size()
        callbacks = [dist.make_allreduce_cb(cls.reduce_mode, dist.WORLD)] if world_size > 1 else None  # noqa
        params_with_grad = cls.params(cfg, model)
        gm.attach(params_with_grad, callbacks=callbacks)
        return gm
