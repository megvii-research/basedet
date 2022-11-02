#!/usr/bin/env python3

import math
import numpy as np

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F

from .hooks import BaseHook

__all__ = ["SyncSizeHook", "YoloxLRSchedulerHook"]


class YoloxLRSchedulerHook(BaseHook):

    def __init__(self, lr, min_lr_ratio, warm_epoch: int = 5, no_aug_epoch: int = 15):
        """
        Args:
            warm_epoch: warm up iters for the frist training epoch.
        """
        self.warm_epoch = warm_epoch
        self.no_aug_epoch = no_aug_epoch
        self.lr = lr
        self.min_lr_ratio = min_lr_ratio
        self.min_lr = self.lr * self.min_lr_ratio

    def before_iter(self):
        lr = self.get_lr()
        pgs = self.trainer.solver.optimizer.param_groups
        # adjust lr for optimzer
        for param_group in pgs:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        trainer = self.trainer

        # epoch_id and iter_id start at 1
        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        max_epoch, max_iter = trainer.progress.max_epoch, trainer.progress.max_iter

        if epoch_id <= self.warm_epoch:
            cur_iter = (epoch_id - 1) * max_iter + iter_id
            lr = (self.lr - self.min_lr) * pow(
                cur_iter / float(self.warm_epoch * max_iter), 2
            ) + self.min_lr
        elif epoch_id > max_epoch - self.no_aug_epoch:
            # last #no_aug_epoch
            lr = self.min_lr
        else:
            cur_iter = (epoch_id - self.warm_epoch) * max_iter + iter_id
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi * (cur_iter - self.warm_epoch * max_iter)
                    / ((max_epoch - self.warm_epoch - self.no_aug_epoch) * max_iter)
                )
            )
        return lr


class SyncSizeHook(BaseHook):

    def __init__(self, multi_size, change_iter=10):
        self.multi_size = multi_size
        self.change_iter = change_iter

    def before_iter(self):
        trainer = self.trainer
        model = trainer.model
        iter_id, max_iter = trainer.progress.iter, trainer.progress.max_iter

        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        max_iter = trainer.progress.max_iter
        cur_iter = (epoch_id - 1) * max_iter + iter_id

        if model.training and cur_iter % self.change_iter == 0:
            if dist.get_rank() == 0:
                size = np.random.choice(self.multi_size)
                tensor = mge.Tensor([size])
            else:
                tensor = mge.Tensor([0])

            if dist.get_world_size() > 1:
                size = F.distributed.broadcast(tensor)
                dist.group_barrier()

            size = int(size)
            model.target_size = (size, size)
