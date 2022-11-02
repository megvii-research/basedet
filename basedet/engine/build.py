#!/usr/bin/env python3
import megfile

import megengine.distributed as dist

from basedet.utils import is_rank0_process, registers, str_timestamp

from .hooks import (
    CheckpointHook,
    EvalHook,
    LoggerHook,
    LRSchedulerHook,
    ResumeHook,
    TensorboardHook
)
from .yolo_hooks import SyncSizeHook, YoloxLRSchedulerHook

__all__ = ["SimpleHookList", "YOLOXHookList"]


@registers.hooks.register()
class SimpleHookList:

    @classmethod
    def build(cls, cfg):
        ckpt_dir = megfile.smart_path_join(cfg.GLOBAL.CKPT_SAVE_DIR, "ckpt")

        hook_list = [LRSchedulerHook()]
        if cfg.TRAINER.RESUME:
            hook_list.append(ResumeHook(ckpt_dir))

        if is_rank0_process():
            if cfg.GLOBAL.TENSORBOARD.ENABLE:
                # Since LoggerHook will reset value, tb hook should be added before LoggerHook
                tb_dir_with_time = megfile.smart_path_join(
                    cfg.GLOBAL.OUTPUT_DIR, "tensorboard", str_timestamp()
                )
                hook_list.append(TensorboardHook(tb_dir_with_time))

            hook_list.append(LoggerHook(cfg.GLOBAL.LOG_INTERVAL))
            hook_list.append(CheckpointHook(ckpt_dir))

        hook_list.append(EvalHook(cfg.TEST.EVAL_EPOCH_INTERVAL))
        return hook_list


@registers.hooks.register()
class YOLOXHookList:

    @classmethod
    def build(cls, cfg):
        ckpt_dir = megfile.smart_path_join(cfg.GLOBAL.CKPT_SAVE_DIR, "ckpt")

        lr = cfg.SOLVER.BASIC_LR * cfg.MODEL.BATCHSIZE * dist.get_world_size()
        hook_list = [
            YoloxLRSchedulerHook(
                lr,
                min_lr_ratio=cfg.SOLVER.MIN_LR_RATIO,
                warm_epoch=cfg.SOLVER.WARM_EPOCH,
                no_aug_epoch=cfg.AUG.TRAIN_SETTING.NO_AUG_EPOCH,
            ),
        ]
        min_range, max_range = cfg.AUG.TRAIN_SETTING.MULTISCALE_RANGE
        ms_size = [x * 32 for x in range(min_range, max_range + 1)]
        hook_list.append(
            SyncSizeHook(
                multi_size=ms_size, change_iter=cfg.AUG.TRAIN_SETTING.SYNC_ITER,
            ),
        )
        if cfg.TRAINER.RESUME:
            hook_list.append(ResumeHook(ckpt_dir))

        if is_rank0_process():
            if cfg.GLOBAL.TENSORBOARD.ENABLE:
                # Since LoggerHook will reset value, tb hook should be added before LoggerHook
                tb_dir_with_time = megfile.smart_path_join(
                    cfg.GLOBAL.OUTPUT_DIR, "tensorboard", str_timestamp()
                )
                hook_list.append(TensorboardHook(tb_dir_with_time))

            hook_list.append(LoggerHook(cfg.GLOBAL.LOG_INTERVAL))
            hook_list.append(CheckpointHook(ckpt_dir))

        hook_list.append(EvalHook(cfg.TEST.EVAL_EPOCH_INTERVAL))
        return hook_list
