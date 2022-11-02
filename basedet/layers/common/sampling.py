#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import megengine.functional as F
from megengine.random import uniform


def sample_labels(labels, num_samples, label_value, ignore_label=-1):
    """
    Sample N labels with label value equal sample value.

    Args:
        labels(Tensor): shape of label is (N,)
        num_samples(int): number of samples.
        label_value(int): sample labels values.

    Returns:
        label(Tensor): label after sampling
    """
    assert labels.ndim == 1, "Only tensor of dim 1 is supported."
    mask = (labels == label_value)
    num_valid = mask.sum()
    if num_valid <= num_samples:
        return labels

    random_tensor = F.zeros_like(labels).astype("float32")
    random_tensor[mask] = uniform(size=num_valid)
    _, invalid_inds = F.topk(random_tensor, k=num_samples - num_valid)

    labels[invalid_inds] = ignore_label
    return labels
