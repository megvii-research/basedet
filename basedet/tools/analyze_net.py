#!/usr/bin/env python3
import argparse
import contextlib
import functools
import importlib
import os
import sys
import numpy as np
from loguru import logger

import megengine as mge
from megengine.utils.module_stats import module_stats
from megengine.utils.profiler import Profiler

from basedet.configs import DetectionConfig
from basedet.utils import all_register, redirect_to_loguru, setup_basedet_logger

from .det_test import get_model_state_dict


@contextlib.contextmanager
def analyzer_adapte(module):
    """
    used for basedet model to fit megengine API.
    """

    def forward(inputs, old_forward):
        *_, height, width = inputs.shape
        img_info = np.array([[height, width, height, width]])
        adaptor_inputs = {"image": inputs, "img_info": img_info}
        return old_forward(adaptor_inputs)

    backup_forward = module.forward
    f = functools.partial(forward, old_forward=backup_forward)
    module.forward = f
    yield module
    module.forward = backup_forward


def default_parser():
    parser = argparse.ArgumentParser(
        description="A script that analyse provided model"
    )
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="training process description file"
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-p", "--profile", action="store_true", help="profile module or not not, default: off",
    )
    parser.add_argument("--height", help="Height of input image.", type=int, default=640)
    parser.add_argument("--width", help="Width of input image.", type=int, default=640)
    parser.add_argument("--channels", help="Width of input image.", type=int, default=3)

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser


def generate_random_input(input_shape):
    *_, height, width = input_shape
    data = mge.random.normal(size=input_shape)
    img_info = np.array([[height, width, height, width]])
    inputs = {"image": data, "img_info": img_info}
    return inputs


def launch_workers(args):
    sys.path.append(os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg: DetectionConfig = current_network.Cfg()
    cfg.merge(args.opts)
    cfg.MODEL.BATCHSIZE = 1
    if args.weight_file is not None:
        cfg.MODEL.WEIGHTS = args.weight_file

    setup_basedet_logger(log_path=cfg.GLOBAL.OUTPUT_DIR, log_file="module_info.txt", to_loguru=True)
    logger.info("args: " + str(args))

    logger.info("Using model named {}".format(cfg.MODEL.NAME))
    model = cfg.build_model()
    model_dict = get_model_state_dict(cfg.MODEL.WEIGHTS)
    if model_dict is not None:
        model.load_state_dict(model_dict)
    model.eval()
    input_size = (1, args.channels, args.height, args.width)
    model_input = generate_random_input(input_size)

    with redirect_to_loguru():
        module_stats(model, model_input)

    if args.profile:
        loops = 20
        logger.info("Profile module with {} loops...".format(loops))
        for _ in range(loops):
            with Profiler("profile"):
                model(model_input)


@logger.catch
def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()
    launch_workers(args)


if __name__ == "__main__":
    main()
