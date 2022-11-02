#!/usr/bin/env python3

import megengine.module as M

from basecls.models import build_model
from basecls.utils import registers

from basedet.configs import ConfigDict
from basedet.layers import feat_storage_to_dict, feature_extract, release_extracted_features_by_key


class BackboneAdapter(M.Module):
    def __init__(self, backbone, in_features):
        super().__init__()
        self.backbone = backbone
        self.in_features = in_features
        self.load_state_dict = self.backbone.load_state_dict
        # self.__dict__.update(backbone.__dict__)

    def extract_features(self, x):
        if self.in_features is None:
            return self(x)

        release_extracted_features_by_key(self.in_features)
        with feature_extract(self.backbone, self.in_features) as storage:
            self.backbone(x)
            feat_dict = feat_storage_to_dict(storage)
        return feat_dict

    def forward(self, x):
        return self.backbone.forward(x)


def build_basecls_backbone(name="resnet50", in_features=None):
    assert name in registers.models.keys(), f"{name} not found in basecls"
    cfg = ConfigDict({"model": {"name": name}})
    _model = build_model(cfg)
    model = BackboneAdapter(_model, in_features)
    return model


def get_feature_setting(backbone_name):
    out_feature_mapper = {
        "resnet": ["s2", "s3", "s4"],
        "regnet": ["s2", "s3", "s4"],
        "snetv2p": ["stage1", "stage2", "stage3"],
        "snet": ["stage0", "stage1", "stage2"],
        "repvgg": ["stage2", "stage3", "stage4"],
        "vgg": ["s3", "s4", "s5"],
        "mbnet": ["s3", "s4", "s5"],
        "effnet": ["s3", "s5", "s7"]
    }
    out_feature = None
    for k, v in out_feature_mapper.items():
        if k in backbone_name:
            out_feature = v
            break
    if out_feature is None:
        raise Exception(
            "cann't find output features of backbone, please set yourself")
    return out_feature


def get_channel_setting(backbone_name):
    out_channel_mapper = {
        # ResNet
        "resnet18": [128, 256, 512],
        "resnet34": [128, 256, 512],
        "resnet50": [512, 1024, 2048],
        "resnet101": [512, 1024, 2048],
        "resnet152": [512, 1024, 2048],
        "resnet18d": [128, 256, 512],
        "resnet34d": [128, 256, 512],
        "resnet50d": [512, 1024, 2048],
        "resnet101d": [512, 1024, 2048],
        "resnet152d": [512, 1024, 2048],
        "se_resnet18": [128, 256, 512],
        "se_resnet34": [128, 256, 512],
        "se_resnet50": [512, 1024, 2048],
        "se_resnet101": [512, 1024, 2048],
        "se_resnet152": [512, 1024, 2048],
        "wide_resnet50_2": [512, 1024, 2048],
        "wide_resnet101_2": [512, 1024, 2048],
        # MobileNet
        "mbnetv1_x025": [64, 128, 256],
        "mbnetv1_x050": [128, 256, 512],
        "mbnetv1_x075": [192, 384, 768],
        "mbnetv1_x100": [256, 512, 1024],
        "mbnetv2_x035": [16, 24, 32],
        "mbnetv2_x050": [16, 32, 48],
        "mbnetv2_x075": [24, 48, 72],
        "mbnetv2_x100": [32, 64, 96],
        "mbnetv2_x140": [48, 88, 136],
        "mbnetv3_small_x075": [192, 120, 432],
        "mbnetv3_small_x100": [240, 144, 576],
        "mbnetv3_large_x075": [96, 64, 528],
        "mbnetv3_large_x100": [120, 80, 672],
        # RegNet
        "regnetx_002": [56, 152, 368],
        "regnetx_004": [64, 160, 384],
        "regnetx_006": [96, 240, 528],
        "regnetx_008": [128, 288, 672],
        "regnetx_016": [168, 408, 912],
        "regnetx_032": [192, 432, 1008],
        "regnetx_040": [240, 560, 1360],
        "regnetx_064": [392, 784, 1624],
        "regnetx_080": [240, 720, 1920],
        "regnetx_120": [448, 896, 2240],
        "regnetx_160": [512, 896, 2048],
        "regnetx_320": [672, 1344, 2520],
        "regnety_002": [56, 152, 368],
        "regnety_004": [104, 208, 440],
        "regnety_006": [112, 256, 608],
        "regnety_008": [128, 320, 768],
        "regnety_016": [120, 336, 888],
        "regnety_032": [216, 576, 1512],
        "regnety_040": [192, 512, 1088],
        "regnety_064": [288, 576, 1296],
        "regnety_080": [448, 896, 2016],
        "regnety_120": [448, 896, 2240],
        "regnety_160": [448, 1232, 3024],
        "regnety_320": [696, 1392, 3712],
        # RegVGG
        "repvgg_a0": [96, 192, 1280],
        "repvgg_a1": [128, 256, 1280],
        "repvgg_a2": [192, 384, 1408],
        "repvgg_b0": [128, 256, 1280],
        "repvgg_b1": [256, 512, 2048],
        "repvgg_b1g2": [256, 512, 2048],
        "repvgg_b1g4": [256, 512, 2048],
        "repvgg_b2": [320, 640, 2560],
        "repvgg_b2g2": [320, 640, 2560],
        "repvgg_b2g4": [320, 640, 2560],
        "repvgg_b3": [384, 768, 2560],
        "repvgg_b3g2": [384, 768, 2560],
        "repvgg_b3g4": [384, 768, 2560],
        "repvgg_d2": [320, 640, 2560],
        # VGG
        "vgg11": [256, 512, 512],
        "vgg11_bn": [256, 512, 512],
        "vgg13": [256, 512, 512],
        "vgg13_bn": [256, 512, 512],
        "vgg16": [256, 512, 512],
        "vgg16_bn": [256, 512, 512],
        "vgg19": [256, 512, 512],
        "vgg19_bn": [256, 512, 512],
        # ShuffleNet
        "snetv2_x050": [24, 48, 96],
        "snetv2_x100": [58, 116, 232],
        "snetv2_x150": [88, 176, 352],
        "snetv2_x200": [122, 244, 488],
        "snetv2p_x075": [52, 104, 208],
        "snetv2p_x100": [64, 128, 256],
        "snetv2p_x125": [84, 168, 336],
        # EfficentNet
        "effnet_b0": [40, 112, 320],
        "effnet_b1": [40, 112, 320],
        "effnet_b2": [48, 120, 352],
        "effnet_b3": [48, 136, 384],
        "effnet_b4": [56, 160, 448],
        "effnet_b5": [64, 176, 512],
        "effnet_b6": [72, 200, 576],
        "effnet_b7": [80, 224, 640],
        "effnet_b8": [88, 248, 704],
        "effnet_l2": [176, 480, 1376],
        "effnet_b0_lite": [40, 112, 320],
        "effnet_b1_lite": [40, 112, 320],
        "effnet_b2_lite": [48, 120, 352],
        "effnet_b3_lite": [48, 136, 384],
        "effnet_b4_lite": [56, 160, 448],
        "effnetv2_s": [64, 160, 256],
        "effnetv2_m": [80, 176, 512],
        "effnetv2_l": [96, 224, 640],
        "effnetv2_b0": [48, 112, 192],
        "effnetv2_b1": [48, 112, 192],
        "effnetv2_b2": [56, 120, 208],
        "effnetv2_b3": [56, 136, 232]
    }

    out_channel = out_channel_mapper.get(backbone_name, None)

    if out_channel is None:
        raise Exception(
            "cann't find output feature num_channels of backbone, please set yourself"
        )
    return out_channel


def get_weights_settting(backbone_name):
    s3_prefix = "s3://basecls/zoo/"
    bucket_prefix = None
    name_list = ["effnet", "regnet", "mbnet", "repvgg", "resnet", "snet", "vgg"]
    for name in name_list:
        if name in backbone_name:
            bucket_prefix = name
            break
    return s3_prefix + f"{bucket_prefix}/{backbone_name}/{backbone_name}.pkl"


def auto_convert_cfg_to_basecls(cfg, basecls_backbone="resnet50"):
    cfg.MODEL.BACKBONE.NAME = "basecls_" + basecls_backbone

    out_feature = get_feature_setting(basecls_backbone)
    out_channels = get_channel_setting(basecls_backbone)
    if out_feature is not None and out_channels is not None:
        cfg.MODEL.BACKBONE.OUT_FEATURES = out_feature
        cfg.MODEL.FPN.TOP_BLOCK_IN_FEATURE = out_feature[-1]
        cfg.MODEL.BACKBONE.OUT_FEATURE_CHANNELS = out_channels
        cfg.MODEL.FPN.TOP_BLOCK_IN_CHANNELS = out_channels[-1]

    cfg.MODEL.WEIGHTS = get_weights_settting(basecls_backbone)
