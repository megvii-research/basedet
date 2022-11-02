#!/usr/bin/env python3
import bisect
import numpy as np

from megengine.data import RandomSampler


class GroupedRandomSampler(RandomSampler):
    """
    Sampler that group sample by given group_ids. Every batched sample has the same group_id.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        group_ids,
        indices=None,
        world_size=None,
        rank=None,
        seed=None,
    ):
        """
        Args:
            dataset (Dataset): dataset used for inference.
            batch_size (int): batch size of output image, default: 1
            group_ids (Sequence): group id of samples.
            indices (Sequence): indice of samples.
            world_size (int): world_size of inference env. default: None, which means total world.
            rank (int): rank id of current process. default: None.
            seed (flaot): sampler seed.
        """
        super().__init__(dataset, batch_size, False, indices, world_size, rank, seed)
        self.group_ids = group_ids
        assert len(group_ids) == len(dataset)
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}

    def batch(self):
        indices = list(self.sample())
        if self.world_size > 1:
            indices = self.scatter(indices)

        batch_index = []
        for ind in indices:
            group_id = self.group_ids[ind]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(ind)
            if len(group_buffer) == self.batch_size:
                batch_index.append(group_buffer)
                self.buffer_per_group[group_id] = []

        return iter(batch_index)

    def __len__(self):
        raise NotImplementedError("length of GroupedRandomSampler is not well-defined.")


class AspectRatioGroupSampler(GroupedRandomSampler):
    """
    Grouping samples from dataset by aspect ratio of images.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        aspect_grouping=[1],
        *args,
        **kwargs,
    ):
        """
        Args:
            dataset (Dataset): dataset used for inference.
            batch_size (int): batch size of output image, default: 1
            aspect_grouping (Sequence): dividing line of aspect grouping.
        """
        def _compute_aspect_ratios(dataset):
            aspect_ratios = []
            for i in range(len(dataset)):
                info = dataset.get_img_info(i)
                aspect_ratios.append(info["height"] / info["width"])
            return aspect_ratios

        def _quantize(x, bins):
            return [bisect.bisect_right(sorted(bins), _) for _ in x]

        if len(aspect_grouping) == 0:
            return RandomSampler(dataset, batch_size, drop_last=True)

        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        super().__init__(dataset, batch_size, group_ids, *args, **kwargs)
