#!/usr/bin/env python3

import unittest

import megengine as mge

from basedet.configs import FasterRCNNConfig
from basedet.models.det import FasterRCNN
from basedet.utils import DummyLoader


class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        self.model = FasterRCNN(FasterRCNNConfig())
        self.loader = DummyLoader()

    @unittest.skipIf(not mge.device.is_cuda_available(), "running on cpu is sooo slow")
    def test_preprocess(self):
        self.model.pre_process(next(self.loader))

    @unittest.skipIf(not mge.device.is_cuda_available(), "running on cpu is sooo slow")
    def test_get_losses(self):
        self.model.train()
        self.model.get_losses(next(self.loader))

    @unittest.skipIf(not mge.device.is_cuda_available(), "running on cpu is sooo slow")
    def test_inference(self):
        self.model.eval()
        self.loader.batch_size = 1
        self.model.inference(next(self.loader))


if __name__ == "__main__":
    unittest.main()
