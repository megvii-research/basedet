#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine.functional as F
from megengine import Tensor


def binary_cross_entropy(
    pred: Tensor, label: Tensor, with_logits: bool = True
) -> Tensor:
    r"""
    Computes the binary cross entropy loss (using logits by default).
    By default(``with_logitis`` is True), ``pred`` is assumed to be logits,
    class probabilities are given by sigmoid.

    Args:
        pred (Tensor): `(N, *)`, where `*` means any number of additional dimensions.
        label (Tensor): `(N, *)`, same shape as the input.
        with_logits (bool): whether to apply sigmoid first. Default: True

    Return:
        loss (Tensor): bce loss value.
    """
    if with_logits:
        # logsigmoid(pred) and logsigmoid(-pred) has common sub-expression
        # hopefully the backend would optimize this
        loss = -(label * F.logsigmoid(pred) + (1 - label) * F.logsigmoid(-pred))
    else:
        loss = -(label * F.log(pred) + (1 - label) * F.log(1 - pred))
    return loss


def weighted_cross_entropy(input, target, weight=None):
    logZ = F.logsumexp(input, axis=1)
    primary_term = F.indexing_one_hot(input, target, axis=1)
    ce_loss = logZ - primary_term
    if weight is not None:
        ce_weight = weight[target.flatten()].reshape(target.shape)
        ce_loss *= ce_weight / ce_weight.mean()
    return ce_loss.mean()
