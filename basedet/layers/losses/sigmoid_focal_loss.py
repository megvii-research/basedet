#!/usr/bin/python3
# -*- coding:utf-8 -*-

import megengine.functional as F
from megengine import Tensor

from .cross_entropy import binary_cross_entropy


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma: float = 0,
) -> Tensor:
    r"""Focal Loss for Dense Object Detection: <https://arxiv.org/pdf/1708.02002.pdf>

    .. math::

        FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)

    Args:
        logits (Tensor): the predicted logits
        targets (Tensor): the assigned targets with the same shape as logits
        alpha (float): parameter to mitigate class imbalance. Default: -1
        gamma (float): parameter to mitigate easy/hard loss imbalance. Default: 0

    Returns:
        the calculated focal loss.
    """
    scores = F.sigmoid(logits)
    loss = binary_cross_entropy(logits, targets, with_logits=True)
    if gamma != 0:
        loss *= (targets * (1 - scores) + (1 - targets) * scores) ** gamma
    if alpha >= 0:
        loss *= targets * alpha + (1 - targets) * (1 - alpha)
    return loss
