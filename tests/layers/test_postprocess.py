#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest

import megengine as mge

from basedet.layers import batched_nms


class PostProcessTest(unittest.TestCase):

    def test_batched_nms(self):
        boxes = mge.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [0.0, 0.0, 100.5, 100.0],
                [0.0, 0.0, 201.0, 200.5],
                [0.0, 0.0, 200.5, 200.5],
                [0.5, 0.5, 100.0, 101.0],
                [0.5, 0.5, 120.5, 120.5],
            ]
        )
        scores = mge.tensor([0.9, 0.8, 0.3, 0.7, 0.6, 0.4])
        labels = mge.tensor([1, 1, 1, 2, 2, 2])
        keep_idx = batched_nms(boxes, scores, labels, iou_thresh=0.4).numpy()
        expected_idx = [0, 3, 4, 2]
        self.assertEqual(list(keep_idx), expected_idx)


if __name__ == '__main__':
    unittest.main()
