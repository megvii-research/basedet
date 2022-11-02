#!/usr/bin/env python3

import argparse
import importlib
import multiprocessing as mp
import os
import sys
import megfile
from loguru import logger

import megengine as mge
import megengine.distributed as dist

from basedet.configs import DetectionConfig
from basedet.utils import all_register, setup_basedet_logger


def default_parser():
    parser = argparse.ArgumentParser(description="A script that train basedet model")
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="training process description file"
    )
    parser.add_argument(
        "-d", "--dir", default=None, type=str, help="training process description file dir"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--ngpus", default=None, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from saved checkpoint or not",
    )
    parser.add_argument(
        "--tensorboard", "--tb", action="store_true", help="use tensorboard or not",
    )
    parser.add_argument(
        "--amp", action="store_true", help="use amp during training or not",
    )
    parser.add_argument(
        "--ema", action="store_true", help="use model ema during training or not",
    )
    parser.add_argument(
        "--dtr", action="store_true",
        help="use dtr during training or not, enable while GPU memory is not enough",
    )
    parser.add_argument(
        "--sync-level", type=int, default=None, help="config sync level, use 0 to debug"
    )
    parser.add_argument(
        "--mp-method", type=str, default="fork", help="mp start method, use fork by defalut"
    )
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument(
        "--debug-mode", action="store_true", help="debug setting, turn on to debug"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser


def launch_workers(args, cfg):
    rank = dist.get_rank()
    logger.info("Init process group for gpu{} done".format(rank))

    cfg.merge(args.opts)

    if args.weight_file is not None:
        cfg.MODEL.WEIGHTS = args.weight_file
    if args.resume:
        cfg.TRAINER.RESUME = True
    if args.amp:
        cfg.TRAINER.AMP.ENABLE = True
    if args.ema:
        cfg.TRAINER.EMA.ENABLE = True
    if args.tensorboard:
        cfg.GLOBAL.TENSORBOARD.ENABLE = True
    if args.debug_mode:
        logger.info("Using debug mode...")
        args.sync_level = 0
        cfg.DATA.NUM_WORKERS = 0

    setup_basedet_logger(log_path=cfg.GLOBAL.OUTPUT_DIR, to_loguru=True)
    logger.info("args: " + str(args))
    if rank == 0:
        logger.info("Create soft link to {}".format(cfg.GLOBAL.OUTPUT_DIR))
        cfg.link_log_dir()

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    if args.dtr:
        logger.info("Using megengine DTR")
        mge.dtr.enable()

    if args.sync_level is not None:
        # NOTE: use sync_level = 0 to debug mge error
        logger.info("Using aysnc_level {}".format(args.sync_level))
        try:
            from megengine.core._imperative_rt.core2 import config_async_level
            config_async_level(args.sync_level)
        except ImportError:
            mge.config.async_level = args.sync_level

    trainer = cfg.build_trainer()
    del cfg
    trainer.train()


@logger.catch(reraise=True)
def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()

    assert args.mp_method in ["fork", "spawn", "forkserver"]
    mp.set_start_method(method=args.mp_method)

    sys.path.append(os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg: DetectionConfig = current_network.Cfg()

    if args.ngpus is None:
        num_devices = mge.device.get_device_count("gpu")
    elif args.ngpus < 0:
        raise ValueError(f"negative device number: {args.ngpus}")
    else:
        num_devices = args.ngpus

    def run():
        if num_devices > 1:
            train = dist.launcher(launch_workers, n_gpus=num_devices)
            train(args, cfg)
        else:
            launch_workers(args, cfg)

    if args.dir:
        root = megfile.SmartPath(args.dir)
        for cfg_file in root.listdir():
            if cfg_file.endswith(".py"):
                args.file = str(root / cfg_file)
                run()
    else:
        run()


if __name__ == "__main__":
    main()
