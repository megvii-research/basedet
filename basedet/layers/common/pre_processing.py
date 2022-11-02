#!/usr/bin/env python3

import megengine.functional as F
from megengine import Tensor

__all__ = [
    "data_to_input", "get_multiple_size", "get_padded_tensor",
]


def data_to_input(image, mean=None, std=None):
    """convert input image to model inputs"""
    image = Tensor(image)
    image = get_padded_tensor(image, 32, 0.0)
    if mean is not None:
        image = image - mean
    if std is not None:
        image = image / std
    return image


def get_multiple_size(input_size: int, multiple: int = 32):
    return (input_size + multiple - 1) // multiple * multiple


def get_padded_tensor(
    tensor: Tensor, multiple_number: int = 32, pad_value: float = 0
) -> Tensor:
    """pad the input tensor to the multiples of multiple_number with given value.

    Args:
        tensor: input tensor, dim of tensor should be greater than 2.
            Last two dim are (height, width).
        multiple_number: make the height and width can be divided by multiple_number.
        pad_value: the value to be padded.

    Returns:
        padded_tensor
    """
    *size, height, width = tensor.shape
    padded_height = get_multiple_size(height, multiple_number)
    padded_width = get_multiple_size(width, multiple_number)

    padded_tensor = F.full(
        (*size, padded_height, padded_width), pad_value, dtype=tensor.dtype
    )

    padded_tensor[..., :height, :width] = tensor
    return padded_tensor
