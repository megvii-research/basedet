#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
import numpy as np

import megengine as mge

from basedet.structures import Boxes


class BoxTest(unittest.TestCase):

    def setUp(self):
        boxes1 = mge.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        self.boxes1 = Boxes(boxes1)

        boxes2 = mge.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.5, 1.5, 1.5],
            ]
        )
        self.boxes2 = Boxes(boxes2)

    def tearDown(self):
        pass

    def test_iou(self):
        expected_ious = np.array(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25 / (2 - 0.25)],
            ]
        )
        ious = self.boxes1.iou(self.boxes2)
        self.assertTrue(np.allclose(ious.numpy(), expected_ious), msg="ious: {}".format(ious))

    def test_ioa(self):
        expected_ioas = np.array(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
            ]
        ).T
        ioas = self.boxes2.ioa(self.boxes1)

        self.assertTrue(np.allclose(ioas.numpy(), expected_ioas), msg="ioas: {}".format(ioas))

    def test_interseaction(self):
        expected_inters = np.array(
            [
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
                [1.0, 0.5, 0.5, 0.25, 0.25, 0.25],
            ]
        )
        inters = self.boxes1.intersection(self.boxes2)

        self.assertTrue(
            np.allclose(inters.numpy(), expected_inters), msg="inters: {}".format(inters)
        )

    def test_scale(self):
        new_boxes = self.boxes1.scale(2, inplace=False)
        self.assertTrue(np.allclose(new_boxes.numpy(), self.boxes1.numpy() * 2))

    def test_center(self):
        centers = self.boxes1.centers
        expected_centers = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )
        self.assertTrue(
            np.allclose(centers.numpy(), expected_centers), msg="centers: {}".format(centers)
        )

    def test_get_item(self):
        sub_box = self.boxes1[:1]
        expected_sub_box = np.array([[0.0, 0.0, 1.0, 1.0]])
        self.assertTrue(np.allclose(sub_box.numpy(), expected_sub_box))
        self.assertIsInstance(sub_box, Boxes)
        value = self.boxes1[0, 0]
        self.assertTrue(int(value.numpy()) == 0)


if __name__ == '__main__':
    unittest.main()
