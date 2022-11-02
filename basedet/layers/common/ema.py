#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.

from copy import deepcopy

import megengine as mge
from megengine.core._imperative_rt.core2 import set_option


def calculate_momentum(
    alpha: float, total_iter: int, update_period: int
):
    """
    pycls style momentum calculation which uses a relative model_ema to decouple momentum with
    other training hyper-parameters e.g.

        * training iters
        * interval to update ema

    Usually the alpha is a tiny positive floating number, e.g. 5e-4,
    with ``max_iter=90000`` and ``update_period=10``, the ema
    momentum should be 0.995, which has roughly same behavior to the default setting.
    i.e. ``momentum=0.9995`` together with ``update_period=1``

    This function is based on the following code of `pycls`:
    https://github.com/facebookresearch/pycls/blob/ee770af5b55cd1959e71af73bf9d5b7d7ac10dc3/pycls/core/net.py#L101-L114  # noqa
    """
    # NOTE: the magic number 90000 comes from 1x coco training total iter number.
    return max(0, 1 - alpha * (90000 * update_period / total_iter))


class ModelEMA:
    """
    Model Exponential Moving Average.
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, momentum, start_iter=0, burnin_iter=2000):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            iters (int): counter of EMA updates.
        """
        # NOTE: EMA model don't need gradient
        self.momentum = momentum
        self.iters = start_iter
        self.burnin_iter = burnin_iter

        self.ema = deepcopy(model)
        self.ema.eval()
        self._ema_states = {k: v for k, v in self.ema.named_parameters()}
        self._ema_states.update({n: p for n, p in self.ema.named_buffers()})

        self._model_states = {k: v for k, v in model.named_parameters()}
        self._model_states.update({n: p for n, p in model.named_buffers()})

    def step(self):
        """
        EMA step with model update.
        """
        self.iters += 1

        if self.iters < self.burnin_iter:
            return
        elif self.iters == self.burnin_iter:
            # momentum = 0 means: ema weight is current model's weight
            self.update(m=0)

        self.update(self.momentum)

    def update(self, m):
        """
        update model with `ema = momentum * ema + (1 - momentum) * model_state_dict`.

        Args:
            m (float): momentum value.
        """
        set_option("record_computing_path", 0)
        for k, v in self._ema_states.items():
            v._reset(v * mge.tensor(m) + mge.tensor(1 - m) * self._model_states[k])
        set_option("record_computing_path", 1)

    def load_state_dict(self, states):
        state_iters = states.get("iter", None)
        if state_iters is not None:
            self.iters = state_iters
        self.ema.load_state_dict(states["model"])

    def state_dict(self):
        return {
            "iter": self.iters,
            "model": self.ema.state_dict(),
        }
