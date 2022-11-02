#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import importlib
import os
import sys
import numpy as np
from loguru import logger

import megengine as mge
from megengine import traced_module as tm

from basedet.utils import all_register, redirect_to_loguru, registers, setup_basedet_logger

from .det_test import get_model_state_dict


def default_parser():
    parser = argparse.ArgumentParser(
        description="A script that dump traced module"
    )
    parser.add_argument(
        "-f", "--file", default="config.py", type=str, help="training process description file",
    )
    parser.add_argument(
        "-o", "--output_path", default="traced_module.pkl", type=str, help="output file name",
    )
    parser.add_argument(
        "-w", "--weight_file", required=True, type=str, help="weights file",
    )
    parser.add_argument(
        "-i", "--input_shape", default=[1, 3, 768, 1280], type=list, nargs=4,
        help="shape of input tensor",
    )

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


def dump_tm(args):
    sys.path.append(os.path.dirname(args.file))
    current_network = importlib.import_module(os.path.basename(args.file).split(".")[0])
    cfg = current_network.Cfg()
    cfg.merge(args.opts)
    cfg.MODEL.BATCHSIZE = 1
    cfg.MODEL.WEIGHTS = args.weight_file

    setup_basedet_logger(log_path=cfg.GLOBAL.OUTPUT_DIR, log_file="dump_tm.txt", to_loguru=True)
    if args.input_shape[0] != 1:
        logger.info("Set batch size to 1 for model inference")
        args.input_shape[0] = 1
    logger.info("args: " + str(args))

    logger.info("Using model named {}".format(cfg.MODEL.NAME))
    model = registers.models.get(cfg.MODEL.NAME)(cfg)
    model_dict = get_model_state_dict(cfg.MODEL.WEIGHTS)
    if model_dict is not None:
        model.load_state_dict(model_dict)
    model.eval()
    model_input = generate_random_input(args.input_shape)

    logger.info("Replace model inference with network_forward")
    model.inference = model.network_forward

    with redirect_to_loguru():
        traced_net = tm.trace_module(model, model_input)
        traced_net.eval()

    logger.info("Dump traced module into {}".format(args.output_path))
    mge.save(traced_net, args.output_path)


def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()
    dump_tm(args)
