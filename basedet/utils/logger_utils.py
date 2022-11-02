#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import sys
from loguru import logger

import megengine.distributed as dist

from basecore.utils import is_rank0_process, setup_mge_logger, str_timestamp


def setup_basedet_logger(log_path, log_file=None, to_loguru=False):
    """
    Args:
        log_path (str): path to save log file.
        log_file (str): log file to save logger infomation. Defalult : train_log.txt
        to_loguru (bool): redirect megengine logger to loguru no not. Default: False
    """
    if log_file is None:
        log_file = "train_log.txt"

    logger.remove()

    filename, suffix = os.path.splitext(log_file)
    if is_rank0_process():
        # logger to stdout/stderr only available on main process
        loguru_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.add(sys.stderr, format=loguru_format)
        time_stamp = str_timestamp()
        logger.add(
            os.path.join(log_path, "{}_{}{}".format(filename, time_stamp, suffix)),
            format=loguru_format,
        )
    dist.group_barrier()
    setup_mge_logger(path=log_path, log_level="INFO", to_loguru=to_loguru)
