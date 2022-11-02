#!/usr/bin/env python3

from basedet import layers, models


def build_backbone(backbone_cfg, **kwargs):
    backbone_name: str = backbone_cfg.get("NAME", "")
    backbone_arch: str = backbone_cfg.get("ARCH", "")
    if backbone_name.startswith("basecls_"):  # build basecls backbone
        from basedet.layers.backbone.product_adapter import build_product_backbone
        # TODO build basecls backbone for research usage
        bottom_up = build_product_backbone(
            backbone_name[len("basecls_"):],
            fpn_in_features=backbone_cfg.OUT_FEATURES,
            **kwargs,
        )
        bottom_up.head = None
        # TODO @ wangfeng02: add norm related and freeze related logic
    elif backbone_arch:
        bottom_up = getattr(models.cls, backbone_arch)(
            block=backbone_cfg.BLOCK,
            config=backbone_cfg,
            norm=layers.get_norm(backbone_cfg.NORM),
            **kwargs,
        )
    else:
        bottom_up = getattr(models.cls, backbone_cfg.NAME)(
            norm=layers.get_norm(backbone_cfg.NORM),
            **kwargs,
        )

    if hasattr(bottom_up, "fc"):
        del bottom_up.fc
    return bottom_up
