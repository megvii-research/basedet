#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import random
import cv2
import numpy as np

from megengine.data.dataset import Dataset

from basedet.data.transforms.yolox_transform import random_affine


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDataset(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        super().__init__()
        self.input_dim = img_size[:2]
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.enable_mosaic = mosaic
        self.mosaic_prob = mosaic_prob

        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob
        self.mixup_scale = mixup_scale

    def __len__(self):
        return len(self._dataset)

    def resize_img(self, items):
        img, boxes, box_labels, img_info = items
        height, width, filename = img_info
        scale = min(self.input_dim[0] / height, self.input_dim[1] / width)

        resize_h, resize_w = int(height * scale), int(width * scale)
        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        boxes *= scale
        img_info = np.array([resize_h, resize_w, height, width, boxes.shape[0]])

        boxes_with_labels = np.concatenate([boxes, box_labels.reshape(-1, 1)], 1)
        return resized_img, boxes_with_labels, img_info, scale, filename

    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            input_h, input_w = self.input_dim

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            mosaic_img = np.full((input_h * 2, input_w * 2, 3), 114, dtype=np.uint8)
            mosaic_gt = []
            for i_mosaic, index in enumerate(indices):
                img, boxes_with_labels, img_info, scale, _ = self.resize_img(self._dataset[index])
                # generate output mosaic image
                h, w, _ = img.shape
                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                mosaic_boxes = boxes_with_labels.copy()
                if mosaic_boxes.size > 0:
                    mosaic_boxes[:, 0] = boxes_with_labels[:, 0] + padw
                    mosaic_boxes[:, 1] = boxes_with_labels[:, 1] + padh
                    mosaic_boxes[:, 2] = boxes_with_labels[:, 2] + padw
                    mosaic_boxes[:, 3] = boxes_with_labels[:, 3] + padh
                mosaic_gt.append(mosaic_boxes)

            if len(mosaic_gt):
                mosaic_gt = np.concatenate(mosaic_gt, 0)
                np.clip(mosaic_gt[:, 0], 0, 2 * input_w, out=mosaic_gt[:, 0])
                np.clip(mosaic_gt[:, 1], 0, 2 * input_h, out=mosaic_gt[:, 1])
                np.clip(mosaic_gt[:, 2], 0, 2 * input_w, out=mosaic_gt[:, 2])
                np.clip(mosaic_gt[:, 3], 0, 2 * input_h, out=mosaic_gt[:, 3])

            mosaic_img, mosaic_gt = random_affine(
                mosaic_img,
                mosaic_gt,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # CopyPaste: https://arxiv.org/abs/2012.07177
            if (
                self.enable_mixup
                and not len(mosaic_gt) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_gt = self.mixup(mosaic_img, mosaic_gt, self.input_dim)
            mix_img, pad_boxes, pad_labels = self.preproc(mosaic_img, mosaic_gt, self.input_dim)
            img_info[-1] = (pad_labels > 0).sum()
            # valid_boxes = pad_boxes[:img_info[-1]]

            return mix_img, pad_boxes, pad_labels, img_info

        else:
            self._dataset._input_dim = self.input_dim
            # img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, labels, img_info, scale, _ = self.resize_img(self._dataset[idx])
            img, boxes, labels = self.preproc(img, labels, self.input_dim)
            return img, boxes, labels, img_info

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        index = random.randint(0, len(self) - 1)
        # img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        img, cp_labels, img_info, scale, _ = self.resize_img(self._dataset[index])

        cp_img = np.ones((*input_dim, 3), dtype=np.uint8) * 114

        cp_img[:img.shape[0], :img.shape[1]] = img

        cp_img = cv2.resize(
            cp_img, (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )

        do_flip = random.uniform(0, 1) > 0.5
        if do_flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), jit_factor, 0, 0, origin_w, origin_h
        )
        if do_flip:
            cp_bboxes_origin_np[:, 0::2] = (origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])

        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
