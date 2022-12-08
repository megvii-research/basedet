#!/usr/bin/env python3

import math

import megengine.module as M

__all__ = [
    "linear_init",
    "linear_weight_init",
    "linear_bias_init",
]


# FIXME: remove this file if megengine fix their Linear init.
def linear_init(module: M.Linear):
    linear_weight_init(module)
    linear_bias_init(module)


def linear_weight_init(module: M.Linear):
    M.init.msra_uniform_(module.weight, a=math.sqrt(5))


def linear_bias_init(module: M.Linear, gain=1.0):
    if module.bias is not None:
        fan_in, _ = M.init.calculate_fan_in_and_fan_out(module.weight)
        bound = gain / math.sqrt(fan_in) if fan_in > 0 else 0
        M.init.uniform_(module.bias, -bound, bound)
