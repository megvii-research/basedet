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
from basedet.data.build import build_test_dataloader
from basedet.engine import BaseTester
from basedet.utils import all_register, is_rank0_process, setup_basedet_logger, unwarp_ckpt


def make_parser():
    parser = argparse.ArgumentParser(description="A script that test basedet model")
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="testing process description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--ngpus", default=None, type=int, help="total number of gpus for testing",
    )
    parser.add_argument(
        "--mp-method", type=str, default="fork", help="mp start method, use fork by defalut"
    )
    parser.add_argument(
        "--ema", action="store_true", help="eval ema model or not",
    )
    return parser


def worker(args):
    sys.path.append(os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])

    cfg: DetectionConfig = current_network.Cfg()
    cfg.MODEL.BATCHSIZE = 1
    output_dir = cfg.GLOBAL.OUTPUT_DIR
    setup_basedet_logger(log_path=output_dir, log_file="test_log.txt", to_loguru=True)
    logger.info("args: " + str(args))
    if is_rank0_process():
        logger.info("Create soft link to {}".format(output_dir))
        cfg.link_log_dir()
        logger.info("Testing config:\n{}".format(cfg))

    if args.weight_file:
        model_file = args.weight_file
    else:
        # if model weights is not given, use last_checkpoint to eval instead.
        ckpt_dir = megfile.smart_path_join(cfg.GLOBAL.CKPT_SAVE_DIR, "ckpt")
        ckpt_file = megfile.smart_path_join(ckpt_dir, "last_checkpoint")
        with megfile.smart_open(ckpt_file, "r") as f:
            last_ckpt = f.read().strip()
        model_file = megfile.smart_path_join(ckpt_dir, last_ckpt)

    model_name = cfg.MODEL.NAME
    logger.info(f"Evaluate model: {model_name}")
    model = cfg.build_model()
    model.load_state_dict(get_model_state_dict(model_file), strict=False)

    evaluator_name = cfg.TEST.EVALUATOR_NAME
    logger.info(f"Evaluator: {evaluator_name}")
    evaluator = cfg.build_evaluator()

    dataloader = build_test_dataloader(cfg)
    tester = BaseTester(model, dataloader, evaluator)
    tester.test()
    if args.ema:
        ema_model = get_model_state_dict(model_file, "ema")
        try:
            model.load_state_dict(ema_model["model"])
        except Exception:
            logger.info("Could not load EMA model")
        else:
            logger.info("Start evaluating EMA model...")
            tester.test()


def get_model_state_dict(model_file, model_key="model"):
    if model_file is None:
        logger.info("No checkpoint founded...")
        return
    logger.info("load model from {}".format(model_file))
    with megfile.smart_open(model_file, "rb") as f:
        state_dict = mge.load(f)
    state_dict = unwarp_ckpt(state_dict, model_key)
    return state_dict


@logger.catch
def main():
    all_register()
    parser = make_parser()
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
        test = dist.launcher(worker, n_gpus=num_devices)
        test(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
