#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import megengine.functional as F

from basedet.layers import get_padded_tensor


class PreProcessTest(unittest.TestCase):

    def test_padded_tensor(self):
        input_shape = [
            (1, 790, 790),
            (1, 799, 799),
            (1, 800, 800),
            (1, 801, 801),
            (2, 10, 630, 630),
            (2, 2, 4, 639, 639),
        ]
        target_shape = [
            (1, 800, 800),
            (1, 800, 800),
            (1, 800, 800),
            (1, 832, 832),
            (2, 10, 640, 640),
            (2, 2, 4, 640, 640),
        ]
        assert len(input_shape) == len(target_shape)
        for input_shape, target_shape in zip(input_shape, target_shape):
            data = F.ones(input_shape)
            pad_tensor = get_padded_tensor(data)
            self.assertTrue(target_shape == pad_tensor.shape)
            self.assertTrue(pad_tensor.sum() == data.sum())


if __name__ == '__main__':
    unittest.main()
