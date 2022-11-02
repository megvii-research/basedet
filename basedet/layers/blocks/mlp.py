#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import megengine.module as M

from basecore.network import get_activation

from basedet import layers


class MLP(M.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_name="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            M.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]
        self.act = get_activation(act_name)
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, M.Linear):
                layers.linear_init(m)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
