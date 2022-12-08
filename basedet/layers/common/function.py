#!/usr/bin/env python3

import numpy as np

import megengine.functional as F

__all__ = [
    "is_empty_tensor", "non_zeros", "permute_to_N_Any_K", "safelog", "meshgrid",
]


def is_empty_tensor(tensor):
    """
    Return True if input tensor is an empty tensor.
    """
    return tensor.size == 0


def non_zeros(tensor):
    """
    Get non zero indices of input tensor.
    """
    return F.cond_take(tensor != 0, tensor)


def permute_to_N_Any_K(tensor, K):
    """
    Transpose and reshape a tensor from (N, C, H, W) to (N, H, W, C) to (N, -1, K)
    """
    assert tensor.ndim == 4
    N = tensor.shape[0]
    return tensor.transpose(0, 2, 3, 1).reshape(N, -1, K)


def safelog(tensor, eps=None):
    """
    Safelog to avoid NaN value by using a tiny eps value.

    Args:
        eps (float): eps value, if not given, decided by tensor dtype.
    """
    if eps is None:
        eps = np.finfo(tensor.dtype).tiny
    return F.log(F.maximum(tensor, eps))


def meshgrid(x, y):
    """meshgrid wrapper for megengine"""
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y
