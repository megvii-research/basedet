#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import datetime
import os
import time
import megfile
from loguru import logger

from tensorboardX import SummaryWriter

from basecore.engine import BaseHook, BaseTester

from basedet.data.build import build_test_dataloader
from basedet.evaluators import build_evaluator
from basedet.utils import (
    Checkpoint,
    MeterBuffer,
    cached_property,
    ensure_dir,
    get_env_info_table,
    get_last_call_deltatime,
    registers
)

__all__ = [
    "BaseHook",
    "LoggerHook",
    "LRSchedulerHook",
    "CheckpointHook",
    "ResumeHook",
    "TensorboardHook",
]


class LoggerHook(BaseHook):
    """
    Hook to log information with logger.

    NOTE: LoggerHook will clear all values in meters, so be careful about the usage.
    """
    def __init__(self, log_interval=20):
        """
        Args:
            log_interval (int): iteration interval between two logs.
        """
        self.log_interval = log_interval
        self.meter = MeterBuffer(self.log_interval)

    def before_train(self):
        logger.info("\nSystem env:\n{}".format(get_env_info_table()))

        # logging model
        logger.info("\nModel structure:\n" + repr(self.trainer.model))

        # logging config
        cfg = self.trainer.cfg
        logger.info("\nTraining full config:\n" + repr(cfg))

        logger.info(
            "Starting training from epoch {}, iteration {}".format(
                self.trainer.progress.epoch, self.trainer.progress.iter
            )
        )
        self.start_training_time = time.time()

    def after_train(self):
        total_training_time = time.time() - self.start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / iter)".format(
                total_time_str, self.meter["iters_time"].global_avg
            )
        )

    def before_iter(self):
        self.iter_start_time = time.time()

    def after_iter(self):
        # TODO wangfeng02: refine logger logic
        single_iter_time = time.time() - self.iter_start_time

        delta_time = get_last_call_deltatime()
        if delta_time is None:
            delta_time = single_iter_time

        self.meter.update({
            "iters_time": single_iter_time,  # to get global average iter time
            "eta_iter_time": delta_time,  # to get ETA time
            "extra_time": delta_time - single_iter_time  # to get extra time
        })

        trainer = self.trainer
        epoch_id, iter_id = trainer.progress.epoch, trainer.progress.iter
        max_epoch, max_iter = trainer.progress.max_epoch, trainer.progress.max_iter

        if iter_id % self.log_interval == 0 or (iter_id == 1 and epoch_id == 1):
            log_str_list = []
            # step info string
            log_str_list.append(str(trainer.progress))

            # loss string
            log_str_list.append(self.get_loss_str(trainer.meter))

            # extra logging meter in model
            extra_str = self.get_extra_str()
            if extra_str:
                log_str_list.append(extra_str)

            # other training info like learning rate.
            log_str_list.append(self.get_train_info_str())

            # memory useage.
            # TODO refine next 3lins logic after mge works
            mem_str = self.get_memory_str(trainer.meter)
            if mem_str:
                log_str_list.append(mem_str)

            # time string
            left_iters = max_iter - iter_id + (max_epoch - epoch_id) * max_iter
            time_str = self.get_time_str(left_iters)
            log_str_list.append(time_str)

            log_str = ", ".join(log_str_list)
            logger.info(log_str)

            # reset meters in trainer & model every #log_interval iters
            trainer.meter.reset()
            if hasattr(trainer.model, "extra_meter"):
                trainer.model.extra_meter.reset()

    def get_loss_str(self, meter):
        """Get loss information during trainging process"""
        loss_dict = meter.get_filtered_meter(filter_key="loss")
        loss_str = ", ".join([
            "{}:{:.3f}({:.3f})".format(name, value.latest, value.avg)
            for name, value in loss_dict.items()
        ])
        return loss_str

    def get_memory_str(self, meter):
        """Get memory information during trainging process"""

        def mem_in_Mb(mem_value):
            return int(mem_value / 1024 / 1024)
        mem_dict = meter.get_filtered_meter(filter_key="memory")
        mem_str = ", ".join([
            "{}:{}({})Mb".format(name, mem_in_Mb(value.latest), mem_in_Mb(value.avg))
            for name, value in mem_dict.items()
        ])
        return mem_str

    def get_train_info_str(self):
        """Get training process related information such as learning rate."""
        # extra info to display, such as learning rate
        trainer = self.trainer
        lr = trainer.solver.optimizer.param_groups[0]["lr"]
        lr_str = "lr:{:.3e}".format(lr)
        return lr_str

    def get_time_str(self, left_iters):
        """Get time related information sucn as data_time, train_time, ETA and so on."""
        trainer = self.trainer
        time_dict = trainer.meter.get_filtered_meter(filter_key="time")
        train_time_str = ", ".join([
            "{}:{:.3f}s".format(name, value.avg)
            for name, value in time_dict.items()
        ])
        # extra time is stored in loggerHook
        train_time_str += ", extra_time:{:.3f}s, ".format(self.meter["extra_time"].avg)

        eta_seconds = self.meter["eta_iter_time"].global_avg * left_iters
        eta_string = "ETA:{}".format(datetime.timedelta(seconds=int(eta_seconds)))
        time_str = train_time_str + eta_string
        return time_str

    def get_extra_str(self):
        """Get extra information provided by model."""
        # extra_meter is defined in BaseNet
        model = self.trainer.model
        extra_str_list = []
        if hasattr(model, "extra_meter"):
            for key, value in model.extra_meter.items():
                if isinstance(value.latest, str):
                    # non-number types like string
                    formatted_str = "{}:{}".format(key, value.latest)
                elif isinstance(value.latest, int):
                    formatted_str = "{}:{}".format(key, value.latest)
                else:
                    formatted_str = "{}:{:.3f}({:.3f})".format(
                        key, float(value.latest), float(value.avg)
                    )
                extra_str_list.append(formatted_str)

        return ", ".join(extra_str_list)


class LRSchedulerHook(BaseHook):
    """
    Hook to adjust solver learning rate.
    """

    def __init__(self, iterwise=True):
        """
        Args:
            warm_iters (int): warm up iters for the frist training epoch.
        """
        self.iterwise = iterwise
        self.scheduler = None

    def before_train(self):
        self.scheduler = self.build_lr_scheduler()

    def before_epoch(self):
        if not self.iterwise:  # epoch wise lr scheduler
            self.scheduler.step(self.trainer.progress)

    def before_iter(self):
        if self.iterwise:
            self.scheduler.step(self.trainer.progress)

    def build_lr_scheduler(self):
        """Build lr scheduler from config, override this method to support more lr scheduler."""
        context = self.trainer
        solver_cfg = context.cfg.SOLVER
        optimizer = context.solver.optimizer
        progress = context.progress

        scheduler_name = solver_cfg.get("LR_SCHEDULER_NAME", "MultiStepLR")

        if scheduler_name == "MultiStepLR":
            milestone = solver_cfg.LR_DECAY_STAGES
            decay_rate = solver_cfg.LR_DECAY_RATE
            if self.iterwise:  # iterwise
                milestone = progress.scale_to_iterwise(milestone)
                logger.info(f"switch to iterwise milestone: {milestone}")
            kwargs = dict(milestones=milestone, gamma=decay_rate)
        else:
            kwargs = dict(solver_cfg.get("EXTRA_LR_ARGS", dict()))

        scheduler = registers.schedulers.get(scheduler_name)(optimizer, **kwargs)

        warm_iters = solver_cfg.get("WARM_ITERS", 0)
        if warm_iters > 0:
            from basecore.engine import WarmUpScheduler
            scheduler = WarmUpScheduler(scheduler, warmup_length=warm_iters)

        return scheduler


class EvalHook(BaseHook):
    """
    Hook to evalutate model during training process.
    """
    def __init__(self, eval_epoch_interval=None):
        self.eval_interval = eval_epoch_interval

    def after_epoch(self):
        if self.eval_interval is not None and self.eval_interval > 0:
            epoch_id, max_epoch = self.trainer.progress.epoch, self.trainer.progress.max_epoch
            if epoch_id != max_epoch and epoch_id % self.eval_interval == 0:
                self.eval()

    def after_train(self):
        self.eval()

    def eval(self):
        # evaluator must know which epoch is evaluated
        self.tester.evaluator.progress = self.trainer.progress
        # train/eval status of model will be auto saved in BaseTester
        self.tester.test()
        self.eval_ema_model()

    def eval_ema_model(self):
        if self.trainer.enable_ema:
            logger.info("Start to evaluate EMA model...")
            ema_model = self.trainer.ema.model
            ema_model.batch_size = 1

            tester = self.tester
            prev_model = tester.model
            tester.model = ema_model
            tester.test()
            self.tester.model = prev_model

    @cached_property
    def tester(self):
        cfg = self.trainer.cfg
        evaluator = build_evaluator(cfg)
        dataloader = build_test_dataloader(cfg)
        model = self.trainer.model
        tester = BaseTester(model, dataloader, evaluator)
        return tester


class CheckpointHook(BaseHook):

    def __init__(self, save_dir=None, save_period=1):
        ensure_dir(save_dir)
        self.save_dir = save_dir
        self.save_period = save_period

    def after_epoch(self):
        trainer = self.trainer
        epoch_id = trainer.progress.epoch

        ckpt_content = {
            "optimizer": trainer.solver.optimizer,
            "progress": trainer.progress,
        }
        if trainer.enable_ema:
            ckpt_content["ema"] = trainer.ema

        ckpt = Checkpoint(self.save_dir, trainer.model, **ckpt_content)
        ckpt.save("latest.pkl")
        logger.info("save checkpoint latest.pkl to {}".format(self.save_dir))

        if epoch_id % self.save_period == 0:
            progress_str = trainer.progress.progress_str_list()
            save_name = "_".join(progress_str[:-1]) + ".pkl"
            ckpt.save(save_name)
            logger.info(f"save checkpoint {save_name} to {self.save_dir}")

    def after_train(self):
        self.trainer.model.dump_weights(os.path.join(self.save_dir, "dumped_model.pkl"))


class ResumeHook(BaseHook):

    def __init__(self, save_dir=None):
        ensure_dir(save_dir)
        self.save_dir = save_dir

    def before_train(self):
        trainer = self.trainer
        model = trainer.model
        progress = trainer.progress
        resume_content = {
            "optimizer": trainer.solver.optimizer,
            "progress": progress,
        }
        if trainer.enable_ema:
            resume_content["ema"] = trainer.ema

        ckpt = Checkpoint(self.save_dir, model, **resume_content)
        filename = ckpt.get_checkpoint_file()
        if megfile.smart_isfile(filename):
            logger.info("load checkpoint from {}".format(filename))
            ckpt.resume()
            # since ckpt is dumped after every epoch,
            # resume training requires epoch + 1 and set iter to 1
            progress.epoch += 1
            progress.iter = 1
        else:
            logger.info(f"checkpoint file {filename} is not found, train from scratch")

        if not trainer.progress.is_first_iter():  # lr_scheduler should be updated
            resume_iter = progress.current_iter()
            trainer.lr_scheduler._step_count = resume_iter
            logger.info(f"resume lr scheduler from value: {resume_iter}")


class TensorboardHook(BaseHook):

    def __init__(self, log_dir, log_interval=20, scalar_type="latest"):
        """
        Args:
            log_dir (str):
            meter_type (str): support values: "latest", "avg", "global_avg", "median"
        """
        assert scalar_type in ("latest", "avg", "global_avg", "median")
        super().__init__()
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.type = scalar_type
        self.log_interval = log_interval

    def create_writer(self):
        return SummaryWriter(self.log_dir)

    def before_train(self):
        self.writer: SummaryWriter = self.create_writer()

    def after_train(self):
        self.writer.close()

    def after_iter(self):
        trainer = self.trainer
        iter_id = trainer.progress.iter
        if iter_id % self.log_interval == 0 or trainer.progress.is_first_iter():
            self.write(context=trainer)

    def write(self, context):
        cur_iter = context.progress.current_iter()
        for key, meter in context.meter.items():
            value = getattr(meter, self.type, None)
            if value is None:
                value = meter.latest
            self.writer.add_scalar(key, value, cur_iter)
        # write lr into tensorboard
        lr = context.solver.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("lr", lr, cur_iter)
