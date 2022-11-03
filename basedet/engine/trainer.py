#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import time

import megengine as mge
import megengine.distributed as dist

from basecore.engine import BaseTrainer, clip_grad

from basedet.layers import ModelEMA, calculate_momentum
from basedet.utils import MeterBuffer, registers


@registers.trainers.register()
class DetTrainer(BaseTrainer):
    """
    Attributes:
        progress: training process. Contains basic informat such as current iter, max iter.
        model: trained model.
        solver: solver that contains optimizer, grad_manager and so on.
        dataloader: data provider.
        meter: meters to log, such as train_time, losses.
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        Args:
            cfg (Config): config which describes training process.
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.dataloader_iter = iter(self.dataloader)
        self.config_to_attr()
        self.meter = MeterBuffer(window_size=self.meter_window_size)  # to store metrics

    def config_to_attr(self):
        self.meter_window_size = self.cfg.GLOBAL.LOG_INTERVAL

        # progress info
        max_epoch = self.cfg.SOLVER.MAX_EPOCH
        num_image_per_epoch = self.cfg.SOLVER.NUM_IMAGE_PER_EPOCH
        if num_image_per_epoch is None:
            # dataloader might be wrapped by InfiniteSampler
            num_image_per_epoch = len(self.dataloader.dataset)

        model_batch_size = self.cfg.MODEL.BATCHSIZE
        max_iter = int(num_image_per_epoch / dist.get_world_size() / model_batch_size)
        self.progress.max_epoch = max_epoch
        self.progress.max_iter = max_iter

        # AMP
        trainer_cfg = self.cfg.TRAINER
        if trainer_cfg.AMP.ENABLE:
            assert self.solver.grad_scaler is not None, "enable AMP but grad_scaler is None"

        # grad clip
        grad_clip_cfg = trainer_cfg.GRAD_CLIP
        if grad_clip_cfg.ENABLE:
            f = clip_grad(self.model.parameters(), grad_clip_cfg.TYPE, **grad_clip_cfg.ARGS)
            self.solver.grad_clip_fn = f

        # model EMA
        ema_config = trainer_cfg.get("EMA", None)
        self.enable_ema = False if ema_config is None else ema_config.ENABLE
        if self.enable_ema:
            momentum = ema_config.MOMENTUM
            if momentum is None:
                total_iter = self.progress.max_epoch * self.progress.max_iter
                update_period = ema_config.UPDATE_PERIOD
                momentum = calculate_momentum(ema_config.ALPHA, total_iter, update_period)
            self.ema = ModelEMA(self.model, momentum, burnin_iter=ema_config.BURNIN_ITER)

    def train_one_iter(self):
        """basic logic of training one iteration."""
        data_tik = time.time()
        model_inputs = next(self.dataloader_iter)
        data_tok = time.time()

        train_tik = time.time()
        loss_dict = self.model_step(model_inputs)
        # TODO calling full_sync in the right way
        mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution
        train_tok = time.time()

        loss_meters = {name: float(loss) for name, loss in loss_dict.items()}
        time_meters = {"train_time": train_tok - train_tik, "data_time": data_tok - data_tik}
        self.meter.update(**loss_meters, **time_meters)

    def model_step(self, model_inputs):
        """
        :meth:`model_step` should be called by :meth:`train_one_iter`, it defines
        basic logic of updating model's parameters.

        Args:
            model_inputs: input of models.
        """
        model_outputs = self.solver.minimize(self.model, model_inputs)
        if self.enable_ema:
            self.ema.step()
        return model_outputs

    @property
    def lr_scheduler(self):
        from .hooks import LRSchedulerHook
        for hook in self._hooks:
            if isinstance(hook, LRSchedulerHook):
                return hook.scheduler
