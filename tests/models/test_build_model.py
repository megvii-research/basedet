#!/usr/bin/env python3

import unittest


class TestBuildModel(unittest.TestCase):

    def setUp(self):
        from basedet.utils import all_register
        all_register()

    def test_build_retinanet(self):
        from basedet.configs import RetinaNetConfig
        RetinaNetConfig().build_model()

    def test_build_freeanchor(self):
        from basedet.configs import FreeAnchorConfig
        FreeAnchorConfig().build_model()

    def test_build_fcos(self):
        from basedet.configs import FCOSConfig
        FCOSConfig().build_model()

    def test_build_atss(self):
        from basedet.configs import ATSSConfig
        ATSSConfig().build_model()

    def test_build_ota(self):
        from basedet.configs import OTAConfig
        OTAConfig().build_model()

    def test_build_yolov3(self):
        from basedet.configs import YOLOv3Config
        YOLOv3Config().build_model()

    def test_build_yolox(self):
        from basedet.configs import YOLOXConfig
        YOLOXConfig().build_model()

    def test_build_faster_rcnn(self):
        from basedet.configs import FasterRCNNConfig
        FasterRCNNConfig().build_model()

    def test_build_centernet(self):
        from basedet.configs import CenterNetConfig
        CenterNetConfig().build_model()

    def test_build_detr(self):
        from basedet.configs import DETRConfig
        DETRConfig().build_model()


if __name__ == "__main__":
    unittest.main()
