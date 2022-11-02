#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import unittest
import numpy as np

import megengine as mge
import megengine.module as M

from basedet.layers import adjust_stats, freeze_norm


class TestLayers(unittest.TestCase):

    def test_adjust_stats(self):
        data = mge.functional.ones((1, 10, 800, 800))
        # use bn since bn changes state during train/val
        model = M.BatchNorm2d(10)
        prev_state = model.state_dict()
        with adjust_stats(model, training=False) as model:
            model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        for k, v in prev_state.items():
            self.assertTrue(all(v == model.state_dict()[k]))

        # test under train mode
        prev_state = model.state_dict()
        with adjust_stats(model, training=True) as model:
            model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        equal_res = [all(v == model.state_dict()[k]) for k, v in prev_state.items()]
        self.assertFalse(all(equal_res))

        # test recurrsive case
        prev_state = model.state_dict()
        with adjust_stats(model, training=False) as model:
            with adjust_stats(model, training=False) as model:
                model(data)
        self.assertTrue(len(model.state_dict()) == len(prev_state))
        for k, v in prev_state.items():
            self.assertTrue(all(v == model.state_dict()[k]))

    def test_freeze_norm(self):
        data = mge.random.normal(size=(1, 10, 800, 800))
        model = M.Sequential(
            M.Conv2d(10, 10, kernel_size=1),
            M.BatchNorm2d(10),
        )
        prev_state = model.state_dict()
        freeze_norm(model)
        model(data)
        for k, v in prev_state.items():
            eqs = (v == model.state_dict()[k])
            if isinstance(eqs, np.ndarray):
                self.assertTrue(eqs.all())
            else:
                self.assertTrue(all(eqs))


if __name__ == '__main__':
    unittest.main()
