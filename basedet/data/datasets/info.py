#!/usr/bin/env python3

from basedet.utils import registers

# INFO include dataset path and dataset meta
INFO = registers.datasets_info

# ==== Predefined datasets and splits for COCO ==========

_COCO_INFO = {}
_COCO_INFO["dataset_type"] = "COCO"
_COCO_INFO["evaluator_type"] = {
    # use different evaluator in different training task.
    "coco": "coco",
    "coco_person": "coco",
}
_COCO_INFO["path"] = {
    # coco 2014
    # format of content: name: (image folder, annotation file)
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    # coco 2017
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
    # coco mini: for megstudio only
    "cocomini_2017_train": ("coco/train", "coco/annotations/cocomini.json"),
}

INFO.register(_COCO_INFO, name="COCO")

# ==== Predefined datasets and splits for Objects365 ==========

_OBJECTS365_INFO = {}
_OBJECTS365_INFO["dataset_type"] = "Objects365"
_OBJECTS365_INFO["evaluator_type"] = {
    # objects365 use the same evaluate logic with COCO datasets.
    "objects365": "coco",
}
_OBJECTS365_INFO["path"] = {
    # format of content: name: (image folder, annotation file)
    "objects365_train":
    ("objects365/train", "objects365/annotations/objects365_train_20190423.json"),
    "objects365_val":
    ("objects365/val", "objects365/annotations/objects365_val_20190423.json"),
    "objects365_test": ("objects365/test", "objects365/annotations/objects365_test_20190423.json"),
    # objects365 tiny version
    "objects365_tiny_train":
    ("objects365/train", "objects365/annotations/objects365_Tiny_train.json"),
    "objects365_tiny_val": ("objects365/val", "objects365/annotations/objects365_Tiny_val.json"),
}

INFO.register(_OBJECTS365_INFO, "OBJECTS365")
