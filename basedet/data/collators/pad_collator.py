#!/usr/bin/env python3

from collections import defaultdict
import numpy as np

from megengine.data import Collator

__all__ = [
    "DetectionPadCollator",
    "DETRPadCollator",
    "calculate_padding_shape",
]


def calculate_padding_shape(original_shape, target_shape):
    assert len(original_shape) == len(target_shape)
    shape = []
    for o, t in zip(original_shape, target_shape):
        shape.append((0, t - o))
    return tuple(shape)


class DetectionPadCollator(Collator):
    """
    Collator used to pad detection images.
    """

    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def apply(self, inputs):
        """
        assume order = ["image", "boxes", "boxes_category", "info"]
        """
        batch_data = defaultdict(list)

        for image, boxes, boxes_category, info in inputs:
            batch_data["data"].append(image.astype(np.float32))
            batch_data["gt_boxes"].append(
                np.concatenate([boxes, boxes_category[:, np.newaxis]], axis=1).astype(np.float32)
            )

            _, current_height, current_width = image.shape
            assert len(boxes) == len(boxes_category)
            num_instances = len(boxes)
            origin_height, origin_width = info[0], info[1]
            info = [current_height, current_width, origin_height, origin_width, num_instances]
            batch_data["im_info"].append(np.array(info, dtype=np.float32))

        for key, value in batch_data.items():
            pad_shape = list(max(s) for s in zip(*[x.shape for x in value]))
            pad_value = [
                np.pad(
                    v, calculate_padding_shape(v.shape, pad_shape), constant_values=self.pad_value
                )
                for v in value
            ]
            batch_data[key] = np.ascontiguousarray(pad_value)

        return batch_data


class DETRPadCollator(DetectionPadCollator):

    def apply(self, inputs):
        batch_data = defaultdict(list)

        for image, boxes, boxes_category, info in inputs:
            batch_data["data"].append(image.astype(np.float32))
            batch_data["gt_boxes"].append(
                np.concatenate([boxes, boxes_category[:, np.newaxis]], axis=1).astype(np.float32)
            )

            _, current_height, current_width = image.shape
            assert len(boxes) == len(boxes_category)
            num_instances = len(boxes)
            origin_height, origin_width = info[0], info[1]
            info = [current_height, current_width, origin_height, origin_width, num_instances]
            batch_data["im_info"].append(np.array(info, dtype=np.float32))

        for key, value in batch_data.items():
            pad_shape = list(max(s) for s in zip(*[x.shape for x in value]))
            pad_value = [
                np.pad(
                    v, calculate_padding_shape(v.shape, pad_shape), constant_values=self.pad_value,
                )
                for v in value
            ]
            if key == "data":
                mask = [
                    np.pad(
                        np.zeros_like(v[0]),
                        calculate_padding_shape(v[0].shape, pad_shape[-2:]),
                        constant_values=1,
                    )
                    for v in value
                ]
                value_mask = [
                    np.concatenate([v, m[np.newaxis, :, :]]) for v, m in zip(pad_value, mask)
                ]
                batch_data[key] = np.ascontiguousarray(value_mask)
            else:
                batch_data[key] = np.ascontiguousarray(pad_value)

        return batch_data
