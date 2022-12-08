#!/usr/bin/env python3


__all__ = ["rename_module", "fuse_model"]


def rename_module(module, replaced_name, new_name):
    """
    Rename submodule in a module to a new name.

    Args:
        module (Module): module which contains submodule.
        replaced_name (str): name of replaced submodule.
        new_name (str): new name of submodule.
    """
    module.__dict__[new_name] = module.__dict__.pop(replaced_name)
    replaced_idx = module._modules.index(replaced_name)
    module._modules[replaced_idx] = new_name


def fuse_model(model):
    """fused conv and bn layer in model"""
    from basecore.network import Conv2d, ConvNormActivation2d, _NORM, fuse_conv_and_bn

    for m in model.modules():
        if isinstance(m, Conv2d) and isinstance(m.norm, _NORM):
            fused_conv = fuse_conv_and_bn(m, m.norm)  # update conv
            m.weight = fused_conv.weight
            m.bias = fused_conv.bias
            m.norm = None
        elif isinstance(m, ConvNormActivation2d) and isinstance(m.norm, _NORM):
            fused_conv = fuse_conv_and_bn(m, m.norm)  # update conv
            m.weight = fused_conv.weight
            m.bias = fused_conv.bias
            m.norm = None
    return model
