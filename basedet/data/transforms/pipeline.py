#!/usr/bin/env python3
from typing import Sequence, Tuple
import numpy as np

from megengine.data.transform.vision.transform import Compose

from basedet.utils import registers


@registers.transforms.register()
class RandomSelect(Compose):
    def __init__(self, transforms=[], batch_compose=False, p=None, **kwargs):
        super().__init__(transforms, batch_compose, **kwargs)
        self.p = p

    def apply_batch(self, inputs: Sequence[Tuple]):
        if self.batch_compose:
            t = np.random.choice(self.transforms, p=self.p)
            return t.apply_batch(inputs)
        else:
            return super().apply_batch(inputs)

    def apply(self, input: Tuple):
        t = np.random.choice(self.transforms, p=self.p)
        return t.apply(input)
