#!/usr/bin/env python3
import math

from megengine.data import MapSampler


class InferenceSampler(MapSampler):
    """
    Sampler for usage of inference process. Available for multi-device also.
    If length of dataset is not divide by world_size exactly, InferenceSampler will
    assign more samples to last device. For example, suppose that dataset
    contains 9 samples and model inferenced on 4 devices, InferenceSampler will
    assign 2 samplers to device0, 1, 2 and 3 sampler to device3.
    """
    def __init__(self, dataset, batch_size=1, world_size=None, rank=None):
        """
        NOTE: in most cases, leave world_size and rank alone.

        Args:
            dataset (Dataset): dataset used for inference.
            batch_size (int): batch size of output image, default: 1
            world_size (int): world_size of inference env. default: None, which means total world.
            rank (int): rank id of current process. default: None.
        """
        super().__init__(dataset, batch_size, False, None, world_size, rank)
        begin = self.num_samples * self.rank
        end = min(self.num_samples * (self.rank + 1), len(self.dataset))
        self.indices = list(range(begin, end))

    def batch(self):
        step, length = self.batch_size, len(self.indices)
        batch_index = [self.indices[i: i + step] for i in range(0, length, step)]
        return iter(batch_index)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))
