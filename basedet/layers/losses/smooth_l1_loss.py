#!/usr/bin/python3
# -*- coding:utf-8 -*-

import megengine.functional as F
from megengine import Tensor


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    r"""Smooth L1 Loss.

    .. math::

        loss_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2 / beta, & \text{if } |x_i - y_i| < beta \\
        |x_i - y_i| - 0.5 * beta, & \text{otherwise }
        \end{cases}

    Args:
        pred (Tensor): the predictions
        target (Tensor): the assigned targets with the same shape as pred
        beta (int): the parameter of smooth l1 loss.

    Returns:
        the calculated smooth l1 loss.
    """
    x = pred - target
    abs_x = F.abs(x)
    if beta < 1e-5:
        loss = abs_x
    else:
        in_loss = 0.5 * x ** 2 / beta
        out_loss = abs_x - 0.5 * beta
        loss = F.where(abs_x < beta, in_loss, out_loss)
    return loss
