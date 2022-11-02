#!/usr/bin/python3
# -*- coding:utf-8 -*-

from typing import List, Tuple

import megengine.functional as F
import megengine.module as M


def get_2d_tuple(x):
    if not isinstance(x, (List, Tuple)):
        return (x, x)
    assert len(x) == 2
    return x


# TODO wangfeng02: refactor caffe pool2d
class CaffePooling2d(M.Module):

    def __init__(self, kernel_size, stride=None, padding=0, mode="max"):
        super().__init__()
        self.kernel_size = get_2d_tuple(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = get_2d_tuple(stride)
        self.padding = get_2d_tuple(padding)
        assert mode in ["max", "average"]
        self.mode = mode

    def forward(self, x):
        ph, pw = self.padding
        kh, kw = self.kernel_size
        sh, sw = self.stride

        h, w = x.shape[2:]

        # Literal rewrite of Caffe logic
        caffe_h = (h + 2 * ph - kh + sh - 1) // sh + 1
        caffe_w = (w + 2 * pw - kw + sw - 1) // sw + 1
        if ph > 0 or pw > 0:
            if (caffe_h - 1) * sh >= h + ph:
                caffe_h = caffe_h - 1
            if (caffe_w - 1) * sw >= w + pw:
                caffe_w = caffe_w - 1

        padding = (ph + sh, pw + sw)
        if self.mode == "max":
            pool = F.max_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw), padding=padding)
        elif self.mode == "average":
            pool = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw), padding=padding)
        else:
            raise NotImplementedError

        pool_h, pool_w = pool.shape[2:]
        h_start = (pool_h - caffe_h) // 2
        w_start = (pool_w - caffe_w) // 2
        pool_h_end = pool_h - h_start
        pool_w_end = pool_w - w_start
        ret = pool[:, :, pool_h_end - caffe_h: pool_h_end, pool_w_end - caffe_w: pool_w_end]
        return ret

    def _module_info_string(self) -> str:
        return "kernel_size={kernel_size}, stride={stride}, padding={padding}, mode={mode}".format(
            **self.__dict__
        )
