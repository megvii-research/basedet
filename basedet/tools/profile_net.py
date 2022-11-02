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
from megengine.utils.profiler import Profiler

from basedet.configs import DetectionConfig
from basedet.engine import DetTrainer
from basedet.utils import all_register, setup_basedet_logger


class DetProfiler(DetTrainer):

    """
    DetProfiler will profile according to network status.
    """
    def train_one_iter(self):
        self.mini_batch = next(self.dataloader_iter)

        model_inputs = {"image": mge.tensor(self.mini_batch["data"])}
        if self.model.training:
            model_inputs.update(
                gt_boxes=mge.tensor(self.mini_batch["gt_boxes"]),
                img_info=mge.tensor(self.mini_batch["im_info"]),
            )
        else:
            if not isinstance(self.mini_batch, dict) or "im_info" not in self.mini_batch:
                img_info = mge.tensor([[800, 800, 800, 800]])
            else:
                img_info = mge.tensor(self.mini_batch["im_info"])
            model_inputs.update(img_info=img_info)

        with Profiler():
            self.model_step(model_inputs)
            mge._full_sync()  # use full_sync func to sync launch queue for dynamic execution

    def model_step(self, model_inputs):
        if self.model.training:
            assert self.solver is not None
            super().model_step(model_inputs)
        else:
            self.model(model_inputs)


def default_parser():
    parser = argparse.ArgumentParser(description="A script that train basedet model")
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="training process description file"
    )
    parser.add_argument(
        "-n", "--ngpus", default=None, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-w", "--weight-file", default=None, type=str, help="weights file",
    )
    parser.add_argument("--iter", default=1, type=int, help="number of profile iters")
    parser.add_argument(
        "--sync-level", type=int, default=None, help="config sync level, use 0 to debug"
    )
    parser.add_argument(
        "--mp-method", type=str, default="fork", help="mp start method, use fork by defalut"
    )
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser


def launch_workers(args):
    rank = dist.get_rank()
    logger.info("Init process group for gpu{} done".format(rank))

    sys.path.append(os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg: DetectionConfig = current_network.Cfg()
    cfg.merge(args.opts)
    cfg.MODEL.WEIGHTS = args.weight_file
    if args.eval:
        cfg.MODEL.BATCHSIZE = 1

    setup_basedet_logger(log_path=cfg.GLOBAL.OUTPUT_DIR, log_file="profile.txt", to_loguru=True)
    logger.info("args: " + str(args))

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    if args.sync_level is not None:
        # NOTE: use sync_level = 0 to debug mge error
        from megengine.core._imperative_rt.core2 import config_async_level
        logger.info("Using aysnc_level {}".format(args.sync_level))
        config_async_level(args.sync_level)

    profiler = build_profiler(cfg, not args.eval)
    profiler.train(start_training_info=(1, args.iter), max_training_info=(1, args.iter))


def build_profiler(cfg, is_model_training=True):
    logger.info("Using model named {}".format(cfg.MODEL.NAME))
    model = cfg.build_model()
    weights = cfg.MODEL.WEIGHTS
    if not weights:
        logger.warning("Profile model without pre-trained weights...")
    else:
        logger.info("Loading model weights from {}".format(weights))
        with megfile.smart_open(weights, "rb") as f:
            model.load_weights(f)

    # sync parameters
    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)
        dist.bcast_list_(model.buffers(), dist.WORLD)

    if is_model_training:
        solver_builder_name = cfg.SOLVER.BUILDER_NAME
        logger.info("Using solver named {}".format(solver_builder_name))
        solver = cfg.build_solver(model)
    else:
        model.eval()
        solver = None

    logger.info("Using dataloader named {}".format(cfg.DATA.BUILDER_NAME))
    dataloader = cfg.build_dataloader()

    return DetProfiler(cfg, model, dataloader, solver)


@logger.catch
def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()

    assert args.mp_method in ["fork", "spawn", "forkserver"]
    mp.set_start_method(method=args.mp_method)

    if args.ngpus is None:
        num_devices = mge.device.get_device_count("gpu")
    elif args.ngpus < 0:
        raise ValueError(f"negative device number: {args.ngpus}")
    else:
        num_devices = args.ngpus

    if num_devices > 1:
        train = dist.launcher(launch_workers, n_gpus=num_devices)
        train(args)
    else:
        launch_workers(args)


if __name__ == "__main__":
    main()
