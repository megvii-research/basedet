#!/usr/bin/env python3
from easydict import EasyDict as edict


class Container(edict):

    def __getitem__(self, idx):
        values = {}
        for k, v in vars(self).items():
            values[k] = v[idx]
        return Container(**values)

    def __str__(self):
        s = self.__class__.__name__ + "("
        s += "data=[{}])".format(", ".join((f"{k}: {v}" for k, v in vars(self).items())))
        return s
