#!/usr/bin/env python3
import math

import megengine.functional as F
import megengine.module as M

from basecore.config import ConfigDict

from basedet.utils import registers


class BasicBlock(M.Module):

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = M.Conv2d(
            in_channels, mid_channels, 3, stride, padding=dilation, bias=False
        )
        self.bn1 = norm(mid_channels)
        self.conv2 = M.Conv2d(mid_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = norm(out_channels)
        self.downsample = (
            M.Identity()
            if in_channels == out_channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, out_channels, 1, stride, bias=False), norm(out_channels),
            )
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x += identity
        x = F.relu(x)
        return x


class Bottleneck(M.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        groups=1,
        base_width=64,
        dilation=1,
        norm=M.BatchNorm2d,
    ):
        super(Bottleneck, self).__init__()
        width = int(mid_channels * (base_width / 64.0)) * groups
        self.conv1 = M.Conv2d(in_channels, width, 1, 1, bias=False)
        self.bn1 = norm(width)
        self.conv2 = M.Conv2d(
            width,
            width,
            3,
            stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm(width)
        self.conv3 = M.Conv2d(width, out_channels, 1, 1, bias=False)
        self.bn3 = norm(out_channels)
        self.downsample = (
            M.Identity()
            if in_channels == out_channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                norm(out_channels),
            )
        )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        identity = self.downsample(identity)

        x += identity
        x = F.relu(x)

        return x


class ResNetV1(M.Module):
    def __init__(
        self,
        config,
        input_channel=None,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm=M.BatchNorm2d,
        head=None,
    ):
        super(ResNetV1, self).__init__()
        block = config.get("block", BasicBlock)
        if isinstance(block, str):
            block = globals()[block]
        if input_channel is not None:
            config.STEM[0]['in_channels'] = input_channel
        self.head = head
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        stem = []
        for stem_conf in config.STEM:
            stem.append(M.Conv2d(**stem_conf))
            stem.append(norm(stem_conf.out_channels))
            stem.append(M.ReLU())
        if config.POOL is not None:
            stem.append(M.MaxPool2d(**config.POOL))
        self.stem = M.Sequential(*stem)

        self._make_layers(block, config, norm)

        if self.head is not None and "w_out" in self.head:
            self.fc = M.Linear(config.LAYERS[-1][2], self.head["w_out"])
        self._init_params(zero_init_residual)
        self.num_stages = len(config.LAYERS)

    def _init_params(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, M.BatchNorm2d):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.msra_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    M.init.uniform_(m.bias, -bound, bound)

        # Zero-initialize the last BN in each residual branch, so that the residual branch
        # starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layers(self, block, config, norm):
        out_channels = config.STEM[-1].out_channels
        for stage_idx, stage in enumerate(config.LAYERS, 2):
            num_blocks = stage[0]
            in_channels = out_channels
            mid_channels = stage[1]
            out_channels = stage[2]
            stride = stage[3]
            dilate = stage[4]

            layer = self._make_layer(
                block,
                num_blocks,
                in_channels,
                mid_channels,
                out_channels,
                stride,
                dilate,
                norm,
            )

            setattr(self, f"layer{stage_idx}", layer)

    def _make_layer(
        self,
        block,
        num_blocks,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        dilate=False,
        norm=M.BatchNorm2d,
    ):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        for layer_idx in range(num_blocks):
            layers.append(
                block(
                    out_channels if layer_idx else in_channels,
                    mid_channels,
                    out_channels,
                    1 if layer_idx else stride,
                    self.groups,
                    self.base_width,
                    self.dilation if layer_idx else previous_dilation,
                    norm,
                )
            )

        return M.Sequential(*layers)

    def extract_features(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x

        for stage_idx in range(2, self.num_stages + 2):
            x = getattr(self, f"layer{stage_idx}")(x)
            outputs[f"s{stage_idx}"] = x
        return outputs

    def forward(self, x):
        x = self.extract_features(x)["s%d" % (self.num_stages + 1)]

        x = F.avg_pool2d(x, 7)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x


__all__ = ["_NAMES"]

_NAMES = [
    # see register_model
]


def register_model(name, MODEL, MODEL_CONFIG):
    config = ConfigDict()
    config.merge(MODEL_CONFIG)

    def model(**kwargs):
        return MODEL(config, **kwargs)
    MODEL_CONFIG['LAYER_ARGS'] = ["NBLOCKS", "MID", "OUT", "STRIDE", "DILATE"]
    model.MODEL_CONFIG = MODEL_CONFIG  # saved it for future reference (in basedet)
    model.__name__ = name
    registers.models.register(model, name)
    _NAMES.append(name)
    return model


# ----------------  -----------
# item              value
# #params           50
# total_param_dims  63.199 K
# total_param_size  246.871 KiB
# total_flops       5.193 MOPs
# total_act_dims    123.696 K
# total_act_size    483.188 KiB
# flops/param_size  20.542
# ----------------  -----------
register_model(
    "resnet_5M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=3,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 8, 8, 2, False],
            [2, 14, 14, 2, False],
            [2, 28, 28, 2, False],
        ]
    )
)


# ----------------  -----------
# item              value
# #params           62
# total_param_dims  100.736 K
# total_param_size  393.500 KiB
# total_flops       9.414 MOPs
# total_act_dims    179.752 K
# total_act_size    702.156 KiB
# flops/param_size  23.364
# ----------------  -----------
register_model(
    "resnet_10M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=8,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 8, 8, 2, False],
            [3, 16, 16, 2, False],
            [3, 32, 32, 2, False],
        ]
    )
)


# ----------------  -----------
# item              value
# #params           62
# total_param_dims  331.312 K
# total_param_size  1.264 MiB
# total_flops       29.515 MOPs
# total_act_dims    308.328 K
# total_act_size    1.176 MiB
# flops/param_size  22.271
# ----------------  -----------
register_model(
    "resnet_30M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=8,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 16, 16, 2, False],
            [3, 32, 32, 2, False],
            [3, 64, 64, 2, False],
        ]
    )
)


# ----------------  -----------
# item              value
# #params           62
# total_param_dims  695.880 K
# total_param_size  2.655 MiB
# total_flops       65.923 MOPs
# total_act_dims    487.080 K
# total_act_size    1.858 MiB
# flops/param_size  23.683
# ----------------  -----------
register_model(
    "resnet_70M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=16,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 24, 24, 2, False],
            [3, 48, 48, 2, False],
            [3, 96, 96, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           62
# total_param_dims  820.872 K
# total_param_size  3.131 MiB
# total_flops       103.226 MOPs
# total_act_dims    694.056 K
# total_act_size    2.648 MiB
# flops/param_size  31.438
# ----------------  ------------
register_model(
    "resnet_100M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 32, 32, 2, False],
            [3, 64, 64, 2, False],
            [3, 96, 96, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           71
# total_param_dims  1.278 M
# total_param_size  4.876 MiB
# total_flops       169.793 MOPs
# total_act_dims    1.569 M
# total_act_size    5.985 MiB
# flops/param_size  33.212
# ----------------  ------------
register_model(
    "resnet_170M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 32, 32, 2, False],
            [4, 64, 64, 2, False],
            [3, 128, 128, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           71
# total_param_dims  2.130 M
# total_param_size  8.126 MiB
# total_flops       259.359 MOPs
# total_act_dims    1.704 M
# total_act_size    6.500 MiB
# flops/param_size  30.437
# ----------------  ------------
register_model(
    "resnet_260M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 32, 32, 2, False],
            [4, 96, 96, 2, False],
            [3, 160, 160, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           77
# total_param_dims  2.899 M
# total_param_size  11.057 MiB
# total_flops       424.271 MOPs
# total_act_dims    2.177 M
# total_act_size    8.306 MiB
# flops/param_size  36.593
# ----------------  ------------
register_model(
    "resnet_420M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [3, 64, 64, 2, False],
            [4, 96, 96, 2, False],
            [3, 192, 192, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           83
# total_param_dims  5.165 M
# total_param_size  19.703 MiB
# total_flops       645.809 MOPs
# total_act_dims    2.435 M
# total_act_size    9.287 MiB
# flops/param_size  31.258
# ----------------  ------------
register_model(
    "resnet_650M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [3, 64, 64, 2, False],
            [5, 128, 128, 2, False],
            [3, 256, 256, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           92
# total_param_dims  5.414 M
# total_param_size  20.654 MiB
# total_flops       823.081 MOPs
# total_act_dims    3.538 M
# total_act_size    13.498 MiB
# flops/param_size  38.004
# ----------------  ------------
register_model(
    "resnet_820M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [1, 32, 32, 1, False],
            [2, 64, 64, 2, False],
            [6, 128, 128, 2, False],
            [3, 256, 256, 2, False],
        ]
    )
)

# ----------------  ----------
# item              value
# #params           98
# total_param_dims  8.718 M
# total_param_size  33.255 MiB
# total_flops       1.188 GOPs
# total_act_dims    3.871 M
# total_act_size    14.766 MiB
# flops/param_size  34.063
# ----------------  ----------
register_model(
    "resnet_1200M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [1, 32, 32, 1, False],
            [2, 64, 64, 2, False],
            [7, 160, 160, 2, False],
            [3, 320, 320, 2, False],
        ]
    )
)

# ----------------  ----------
# item              value
# #params           161
# total_param_dims  25.557 M
# total_param_size  97.492 MiB
# total_flops       4.169 GOPs
# total_act_dims    22.430 M
# total_act_size    85.562 MiB
# flops/param_size  40.779
# ----------------  ---------- #from basedet resnet50_v1
register_model(
    "resnet_4200M",
    ResNetV1,
    dict(
        block="Bottleneck",
        STEM=[
            dict(
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
        ],
        POOL=dict(
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [3, 64, 256, 1, False],
            [4, 128, 512, 2, False],
            [6, 256, 1024, 2, False],
            [3, 512, 2048, 2, False]
        ]
    )
)

# ========= 以下为来自业务老backbone =========

# ----------------  ------------
# item              value
# #params           78
# total_param_dims  1.197 M
# total_param_size  4.568 MiB
# total_flops       138.960 MOPs
# total_act_dims    1.243 M
# total_act_size    4.741 MiB
# flops/param_size  29.014
# ----------------  ------------
# https://git-core.megvii-inc.com/base-detection/basedet/-/blob/master/playground/supersafety/smoke.atss/config.py#L57-72
register_model(
    "resnet_139M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
        ],
        POOL=dict(
            kernel_size=2,
            stride=2,
            padding=0,
        ),
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 16, 16, 1, False],
            [2, 32, 32, 2, False],
            [3, 64, 64, 2, False],
            [3, 128, 128, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           93
# total_param_dims  1.587 M
# total_param_size  6.055 MiB
# total_flops       186.338 MOPs
# total_act_dims    1.193 M
# total_act_size    4.550 MiB
# flops/param_size  29.348
# ----------------  ------------
# https://git-core.megvii-inc.com/lizeming/basedet/-/blob/gw/sup_safety/basedet/model/cls/resnet/edge/res187m.py
register_model(
    "resnet_186M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=16,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=True
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 16, 16, 1, False],
            [3, 32, 32, 2, False],
            [4, 64, 64, 2, False],
            [4, 128, 128, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           117
# total_param_dims  2.976 M
# total_param_size  11.354 MiB
# total_flops       367.883 MOPs
# total_act_dims    2.008 M
# total_act_size    7.660 MiB
# flops/param_size  30.900
# ----------------  ------------
# https://git-core.megvii-inc.com/gd_products/security_face/tree/smy/models/face_retinanet/shangmingyang/config/3516DV300.face_binding_person.gray.resnet_326m.same_anchor.ATSS.refinehumandata.376M-k5s4p2_oc24-s3455.0.6center.3life.128batch
register_model(
    "resnet_368M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=24,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=True
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [3, 16, 24, 1, False],
            [4, 32, 48, 2, False],
            [5, 64, 96, 2, False],
            [5, 128, 192, 2, False],
        ]
    )
)

# ----------------  ------------
# item              value
# #params           90
# total_param_dims  3.590 M
# total_param_size  13.695 MiB
# total_flops       453.101 MOPs
# total_act_dims    2.704 M
# total_act_size    10.316 MiB
# flops/param_size  31.552
# ----------------  ------------
# https://git-core.megvii-inc.com/lizeming/basedet/-/blob/gw/sup_safety/basedet/model/cls/resnet/server/res542m.py
register_model(
    "resnet_453M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=32,
                kernel_size=5,
                stride=4,
                padding=2,
                bias=True
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [2, 24, 48, 1, False],
            [3, 24, 96, 2, False],
            [3, 48, 192, 2, False],
            [4, 96, 384, 2, False],
        ]
    )
)

# ----------------  ----------
# item              value
# #params           124
# total_param_dims  5.735 M
# total_param_size  21.879 MiB
# total_flops       1.063 GOPs
# total_act_dims    3.940 M
# total_act_size    15.029 MiB
# flops/param_size  46.349
# ----------------  ----------
# https://git-core.megvii-inc.com/base-detection/basedet/-/blob/vdet_product/playground/vdet_product/6in1detection.res991m.atss/basemodel.py
register_model(
    "resnet_1063M",
    ResNetV1,
    dict(
        STEM=[
            dict(
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            dict(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        ],
        POOL=None,
        LAYERS=[
            # num_blocks, mid_channels, out_channels, stride, dilate
            [3, 32, 32, 1, False],
            [6, 64, 64, 2, False],
            [6, 128, 128, 2, False],
            [3, 256, 256, 2, False],
        ]
    )
)
