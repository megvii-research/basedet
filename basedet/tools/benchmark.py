#!/usr/bin/env python3

import argparse
import concurrent.futures
import functools
import json
import multiprocessing as mp
import subprocess
import time
from typing import List
from loguru import logger

import megengine as mge
import megengine.distributed as dist

from basecore.utils import get_command_path, get_device_name

from basedet.configs import DetectionConfig
from basedet.utils import DummyLoader, MeterBuffer, all_register

_QUIT_SIGNAL = False


def quit_signal():
    return _QUIT_SIGNAL


def set_quit_signal():
    global _QUIT_SIGNAL
    _QUIT_SIGNAL = True


def reset_quit_signal():
    global _QUIT_SIGNAL
    _QUIT_SIGNAL = False


def default_parser():
    parser = argparse.ArgumentParser(description="A script that benchmark basedet model")
    parser.add_argument(
        "-n", "--ngpus", default=None, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-o", "--output", default="bench.json", type=str, help="filename to save all results",
    )
    parser.add_argument("--amp", action="store_true", help="use amp or not")
    parser.add_argument(
        "--dtr", action="store_true",
        help="use dtr during training or not, enable while GPU memory is not enough",
    )
    parser.add_argument("--eval", action="store_true", help="apply evaluation or not")
    parser.add_argument("--valid-input", action="store_true", help="use valid input or not")
    parser.add_argument(
        "--mp-method", type=str, default="fork", help="mp start method, use fork by defalut"
    )
    parser.add_argument("--fastrun", action="store_true")
    parser.add_argument("--warm-iters", default=20, type=int)
    parser.add_argument("--total-iters", default=100, type=int)
    return parser


def parse_nvcc_output(output_string: str):
    lines = output_string.strip().splitlines()
    lines = [line.split(",") for line in lines]
    exec_info = {}
    for i, info in enumerate(lines[0]):
        sum_value = sum([int(v[i].split()[0]) for v in lines[1:]])
        exec_info[info.split()[0]] = sum_value
    return exec_info


def command_worker(cmd_string, sleep_time=0.5):
    value_queue = {}
    while True:
        cmd_output = subprocess.check_output(cmd_string, shell=True).decode("utf-8")
        exec_info = parse_nvcc_output(cmd_output)
        if "utilization.gpu" not in exec_info or exec_info["utilization.gpu"] > 0:
            # ensure that gpu is running
            for k, v in exec_info.items():
                if k in value_queue:
                    value_queue[k].append(v)
                else:
                    value_queue[k] = [v]
        time.sleep(sleep_time)
        if quit_signal():
            return value_queue


def gpu_monitor(watch_time=0.5):

    def wrap(f):

        @functools.wraps(f)
        def inner_func(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                nvidia_smi = get_command_path("nvidia-smi")
                queries = ["utilization.gpu", "memory.used"]
                query_str = ",".join(queries)
                cmd_string = f"{nvidia_smi} --query-gpu={query_str} --format=csv"
                future = executor.submit(command_worker, cmd_string, sleep_time=watch_time)

                func_ret = f(*args, **kwargs)
                set_quit_signal()
                res = future.result()
                reset_quit_signal()

                return func_ret, res

        return inner_func

    return wrap


class Benchmark:

    def __init__(self, model, dataloader, bench_cfg):
        self.model = model
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.cfg = bench_cfg
        self.warm_iters = bench_cfg.warm_iters
        self.total_iters = bench_cfg.total_iters
        self.meter = MeterBuffer(window_size=self.total_iters)

    def loop(self):
        for cur_iter in range(self.total_iters):
            model_input = next(self.dataloader_iter)
            mge._full_sync()
            tic = time.perf_counter()
            self.model_step(model_input)
            mge._full_sync()
            tok = time.perf_counter()
            if (cur_iter + 1) % 10 == 0:
                logger.info(f"Benchmark progress: {cur_iter + 1} / {self.total_iters}")
            if cur_iter > self.warm_iters:
                self.meter.update({"iter_time": tok - tic})
            else:
                self.meter.update({"warm_iter_time": tok - tic})

        return self.dump_benchmark_data()

    def model_step(self, model_input):
        raise NotImplementedError

    def dump_benchmark_data(self):
        return self.meter["iter_time"]


class TrainBenchmark(Benchmark):

    def __init__(self, *args, solver, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = solver

    def model_step(self, model_input):
        self.solver.minimize(self.model, model_input)


class EvalBenchmark(Benchmark):

    def model_step(self, model_input):
        model_input = {k: v[:1, ...] for k, v in model_input.items()}
        self.model(model_input)


def benchmark_one_model(cfg, args):
    model_name = cfg.MODEL.NAME
    logger.info(f"benchmark model name: {model_name}")
    model = cfg.build_model()
    if args.valid_input:
        dataloader = cfg.build_dataloader()
    else:
        dataloader = DummyLoader()

    if args.eval:
        logger.info("Eval benchmark")
        model.eval()
        model.batch_size = 1
        benchmark = EvalBenchmark(model, dataloader, args)
    else:
        logger.info("Train benchmark")
        solver = cfg.build_solver(model)
        model.train()
        benchmark = TrainBenchmark(model, dataloader, args, solver=solver)

    time_bench = benchmark.loop()
    bench_results = {"iter_time": time_bench.avg}
    logger.info("benchmark gpu memory usage now...")
    # monkey patch `Benchmark.loop` method
    benchmark.loop = gpu_monitor(watch_time=0.5)(benchmark.loop)
    _, gpu_bench = benchmark.loop()
    for k, v in gpu_bench.items():
        v = v[3:]  # remove top3 to skip warmup recording at the begining, might tune in the future
        bench_results[k] = sum(v) / len(v)
    return {model_name: bench_results}


def modify_cfg(cfg):
    for aug in cfg.AUG.TRAIN_VALUE:
        if aug[0] == "MGE_ShortestEdgeResize":
            aug[1]["min_size"] = 800
            aug[1]["max_size"] = 800


def generate_json_tag(args):
    tag = {}
    tag["world_size"] = dist.get_world_size()
    tag["amp"] = args.amp
    tag["fastrun"] = args.fastrun
    tag["eval mode"] = args.eval
    tag["dummy input"] = not args.valid_input
    tag["device"] = get_device_name()
    return tag


def benchmark_all_models(args):
    from basedet.configs import (
        ATSSConfig,
        # FasterRCNNConfig,
        FCOSConfig,
        FreeAnchorConfig,
        RetinaNetConfig,
    )
    if dist.get_rank() != 0:
        logger.remove()

    if args.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    cfg_list: List[DetectionConfig] = [
        # FasterRCNNConfig(),
        RetinaNetConfig(),
        ATSSConfig(),
        FCOSConfig(),
        FreeAnchorConfig(),
    ]
    dumped_dict = {}
    for cfg in cfg_list:
        if args.amp:
            cfg.TRAINER.AMP.ENABLE = True
        modify_cfg(cfg)  # resize to 800/1333 config
        bench_result = benchmark_one_model(cfg, args)
        dumped_dict.update(bench_result)
    dumped_dict["tag"] = generate_json_tag(args)
    with open(args.output, "w") as f:
        json.dump(dumped_dict, f, indent=4)
        logger.info(f"write json file to {args.output}")


@logger.catch
def main():
    all_register()
    parser = default_parser()
    args = parser.parse_args()
    logger.info(f"args: {args}")

    assert args.mp_method in ["fork", "spawn", "forkserver"]
    mp.set_start_method(method=args.mp_method)

    if args.ngpus is None:
        num_devices = mge.device.get_device_count("gpu")
    elif args.ngpus < 0:
        raise ValueError(f"negative device number: {args.ngpus}")
    else:
        num_devices = args.ngpus

    if num_devices > 1:
        train = dist.launcher(benchmark_all_models, n_gpus=num_devices)
        train(args)
    else:
        benchmark_all_models(args)


if __name__ == "__main__":
    main()
