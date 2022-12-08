#!/usr/bin/env python3

import megengine as mge
from megengine import module as M

__all__ = ["DropPath", "linear_drop_prob"]


class DropPath(M.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # it's equal to use dropout, but following code is 10% faster
        mask = mge.random.uniform(size=shape) < keep_prob
        output = (x / keep_prob) * mask
        return output

    def _module_info_string(self) -> str:
        return "drop_prob={drop_prob}".format(**self.__dict__)


def linear_drop_prob(depth, last_prob):
    """
    Args:
        depth (List): depth list of different layers.
        last_prob (float): drop prob of last layer.
    """
    prob_iter = (i / sum(depth) * last_prob for i in range(sum(depth)))
    drop_prob_list = [[next(prob_iter) for _ in range(d)] for d in depth]
    return drop_prob_list
