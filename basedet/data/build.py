#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. All rights reserved.

import copy
import os
from loguru import logger

from megengine.data import DataLoader, Infinite

from basedet.utils import registers

from .collators import DetectionPadCollator, DETRPadCollator
from .samplers import AspectRatioGroupSampler, InferenceSampler

__all__ = [
    "build_transform",
    "build_dataset",
    "build_test_dataloader",
    "get_basedet_data_dir",
    "get_images_dir_and_anno_path",
]


def build_transform(cfg, mode="train"):
    """
    Build transforms according to given cfg for train/inference process.

    Args:
        cfg : config of transforms.
        mode (str): transform mode, support "train" and "test" only.
    """
    transforms = []
    assert mode in ["train", "test"], "{} mode not supported".format(mode)
    if mode == "train":
        for name, kwargs in cfg.AUG.TRAIN_VALUE:
            transforms.append(registers.transforms.get(name)(**kwargs))

        for name, kwargs in cfg.AUG.TRAIN_WRAPPER:
            transforms = registers.transforms.get(name)(transforms, **kwargs)
    elif mode == "test":
        for name, kwargs in cfg.TEST.AUG.VALUE:
            transforms.append(registers.transforms.get(name)(**kwargs))
        for name, kwargs in cfg.TEST.AUG.WRAPPER:
            transforms = registers.transforms.get(name)(transforms, **kwargs)

    return transforms


def get_basedet_data_dir():
    """
    Get dataset dataset dir path of basedet. dataset dir value could be set by
    OS envriment $BASEDET_DATA_DIR, if not set, use `datasets` directory under
    dirname(basedet.__file__) instead.
    """
    import basedet
    basedet_root = os.path.dirname(os.path.dirname(basedet.__file__))
    data_dir = os.path.join(basedet_root, "datasets")
    return os.environ.get("BASEDET_DATA_DIR", data_dir)


def get_images_dir_and_anno_path(dataset_fullname):
    """
    Get annotation and images dir from full named dataset.
    Dataset name might looks like "coco_train_2017", "objects365_train".

    Args:
        dataset_fullname (str): dataset name.
    """
    dataset_name = dataset_fullname.split("_")[0].upper()
    assert dataset_name in registers.datasets_info, "{} is not found.".format(dataset_name)
    dataset_info = registers.datasets_info.get(dataset_name)

    images_dir, anno_path = dataset_info["path"][dataset_fullname]
    data_dir = get_basedet_data_dir()
    images_dir = os.path.join(data_dir, images_dir)
    anno_path = os.path.join(data_dir, anno_path)
    return images_dir, anno_path


def build_dataset(cfg, mode="train"):
    """
    Build dataset according to given cfg for train/inference process.

    Args:
        cfg : config of dataset.
        mode (str): transform mode, support "train" and "test" only.
    """
    assert mode in ["train", "test"], "{} mode not supported".format(mode)
    if mode == "train":
        dataset_args = copy.deepcopy(cfg.DATA.TRAIN)  # since name will be poped, deepcopy here
    elif mode == "test":
        dataset_args = copy.deepcopy(cfg.DATA.TEST)  # since name will be poped, deepcopy here
    else:
        raise ValueError("unrecognized mode {}".format(mode))

    dataset_full_name = dataset_args.pop("name")
    images_dir, anno_path = get_images_dir_and_anno_path(dataset_full_name)
    dataset_name = dataset_full_name.split("_")[0].upper()
    dataset_class_name = registers.datasets_info.get(dataset_name)["dataset_type"]

    dataset_class = registers.datasets.get(dataset_class_name)
    logger.info("Using dataset named: {}".format(dataset_class_name))
    logger.info("Dataset images_path: {}, anno_path: {}".format(images_dir, anno_path))
    logger.info("Dataset extra args: {}".format(dataset_args.to_dict()))
    return dataset_class(images_dir, anno_path, **dataset_args)


def build_test_dataloader(cfg):
    """
    Build test dataloader according to cfg value, since InferenceSampler is nearly the only choice,
    no dataloader builder for test process.
    """
    val_dataset = build_dataset(cfg, "test")
    val_sampler = InferenceSampler(val_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, num_workers=2)
    return val_dataloader


def build_dataloader(cfg, dataset=None, transform=None, sampler=None, collator=None):
    if dataset is None:
        dataset = build_dataset(cfg)

    if transform is None:
        transform = build_transform(cfg)

    if sampler is None:
        batch_size = cfg.MODEL.BATCHSIZE
        sampler = AspectRatioGroupSampler(dataset, batch_size)
        if cfg.DATA.ENABLE_INFINITE_SAMPLER:
            sampler = Infinite(sampler)

    if collator is None:
        collator = DetectionPadCollator()

    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        transform=transform,
        collator=collator,
        num_workers=cfg.DATA.NUM_WORKERS,
    )


@registers.dataloader.register()
class DataloaderBuilder:
    """
    Builder of BaseDet dataloader, core logic of building is implemented in :meth:`build` function.
    For custom usage of sampler and collator, override :meth:`build_sampler`
    and :meth:`build_collator` if neccessry.
    """

    @classmethod
    def build(cls, cfg):
        dataset = build_dataset(cfg)
        dataloader = build_dataloader(
            cfg,
            dataset=dataset,
            sampler=cls.build_sampler(dataset, cfg),
            transform=build_transform(cfg),
            collator=cls.build_collator(cfg),
        )
        return dataloader

    @classmethod
    def build_sampler(cls, dataset, cfg):
        # for compatibility with old dataloader
        batch_size = cfg.MODEL.BATCHSIZE
        sampler = AspectRatioGroupSampler(dataset, batch_size)
        if cfg.DATA.ENABLE_INFINITE_SAMPLER:
            sampler = Infinite(sampler)
        return sampler

    @classmethod
    def build_collator(cls, cfg):
        # for compatibility with old dataloader
        return DetectionPadCollator()


@registers.dataloader.register()
class DETRDataloaderBuilder(DataloaderBuilder):

    @classmethod
    def build(cls, cfg):
        return build_dataloader(cfg, collator=DETRPadCollator())


@registers.dataloader.register()
class YOLOXDataloaderBuilder:
    """
    Builder of BaseDet dataloader, core logic of building is implemented in :meth:`build` function.
    For custom usage of sampler and collator, override :meth:`build_sampler`
    and :meth:`build_collator` if neccessry.
    """

    @classmethod
    def build(cls, cfg):
        from basedet.data.datasets.mosaic_dataset import MosaicDataset
        from basedet.data.transforms.yolox_transform import TrainTransform
        from megengine.data.sampler import RandomSampler

        aug_cfg = cfg.AUG.TRAIN_SETTING

        inner_dataset = build_dataset(cfg)
        mosaic_dataset = MosaicDataset(
            dataset=inner_dataset,
            img_size=aug_cfg.INPUT_SIZE,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=aug_cfg.FLIP_PROB,
                hsv_prob=aug_cfg.HSV_PROB,
            ),
            degrees=aug_cfg.DEGREES,
            translate=aug_cfg.TRANSLATE,
            mosaic_scale=aug_cfg.MOSAIC_SCALE,
            mixup_scale=aug_cfg.MIXUP_SCALE,
            shear=aug_cfg.SHEAR,
            enable_mixup=aug_cfg.ENABLE_MIXUP,
            mosaic_prob=aug_cfg.MOSAIC_PROB,
            mixup_prob=aug_cfg.MIXUP_PROB,
        )
        train_dataloader = DataLoader(
            mosaic_dataset,
            sampler=Infinite(RandomSampler(mosaic_dataset, batch_size=cfg.MODEL.BATCHSIZE)),
            num_workers=cfg.DATA.NUM_WORKERS,
            collator=DetectionPadCollator(),
        )
        return train_dataloader
