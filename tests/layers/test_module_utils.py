#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import megengine as mge
import megengine.module as M

from basedet.layers import Conv2d, fuse_model, rename_module


class ModuleForTest(M.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 10, 3, norm="BN", activation="relu")

    def forward(self, x):
        return self.conv1(x)

    def rename_forward(self, x):
        return self.conv2(x)


class ModuleUtilsTest(unittest.TestCase):

    def test_rename_module(self):
        module = ModuleForTest()
        module.eval()

        data = mge.random.normal(size=(1, 3, 10, 10))
        output1 = module(data)
        rename_module(module, "conv1", "conv2")
        m = getattr(module, "conv2", None)
        self.assertTrue(m is not None)
        output2 = module.rename_forward(data)
        val = (output1 != output2).sum().item()
        self.assertEqual(val, 0)

    def test_fuse_module(self):
        conv = Conv2d(3, 3, 1, norm="BN")
        conv.norm.eps = 0  # avoid eps caused precision lost
        conv.eval()

        data = mge.random.normal(size=(1, 3, 10, 10))
        output1 = conv(data)
        fused_conv = fuse_model(conv)
        output2 = fused_conv(data)
        val = (output1 != output2).sum().item()
        self.assertEqual(val, 0)


if __name__ == '__main__':
    unittest.main()
