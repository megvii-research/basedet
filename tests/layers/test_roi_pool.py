#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
import numpy as np

import megengine as mge
import megengine.functional as F

from basedet.layers import roi_pool


class RoIPoolTest(unittest.TestCase):

    @unittest.skipIf(not mge.device.is_cuda_available(), "test cuda only")
    def setUp(self):
        H, W = 5, 5
        feature = F.arange(H * W).reshape(1, 1, H, W).astype("float32")
        """
        0  1  2   3 4
        5  6  7   8 9
        10 11 12 13 14
        15 16 17 18 19
        20 21 22 23 24
        """
        self.feat = feature

    def tearDown(self):
        pass

    @unittest.skipIf(not mge.device.is_cuda_available(), "test cuda only")
    def test_roi_align(self):
        rois = mge.tensor([[0, 1, 1, 3, 3]])
        pool_shape = 4

        align_results = np.array([
            [4.5, 5.0, 5.5, 6.0],
            [7.0, 7.5, 8.0, 8.5],
            [9.5, 10.0, 10.5, 11.0],
            [12.0, 12.5, 13.0, 13.5],
        ])
        align_output = roi_pool(
            [self.feat], rois, strides=[1], pool_shape=pool_shape, pooler_type="roi_align"
        )
        self.assertTrue(np.allclose(align_output.numpy(), align_results))

    @unittest.skipIf(not mge.device.is_cuda_available(), "test cuda only")
    def test_roi_pool(self):
        rois = mge.tensor([[0, 1, 1, 3, 3]])
        pool_shape = 4

        pool_results = np.array([
            [6.0, 7.0, 8.0, 8.0],
            [11.0, 12.0, 13.0, 13.0],
            [16.0, 17.0, 18.0, 18.0],
            [16.0, 17.0, 18.0, 18.0],
        ])
        pool_output = roi_pool(
            [self.feat], rois, strides=[1], pool_shape=pool_shape, pooler_type="roi_pool"
        )
        self.assertTrue(np.allclose(pool_output.numpy(), pool_results))

    @unittest.skipIf(not mge.device.is_cuda_available(), "test cuda only")
    def test_resize(self):
        rois = mge.tensor([[0, 1, 1, 3, 3]])
        pool_shape = 4

        output = roi_pool(
            [self.feat], rois, strides=[1], pool_shape=pool_shape, pooler_type="roi_align"
        )
        feat2x = F.vision.interpolate(self.feat, scale_factor=2)
        output2x = roi_pool(
            [feat2x], rois, strides=[1 / 2], pool_shape=pool_shape, pooler_type="roi_align"
        )
        self.assertTrue(np.allclose(output2x.numpy(), output.numpy()))


if __name__ == '__main__':
    unittest.main()
