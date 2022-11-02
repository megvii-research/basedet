#!/usr/bin/env python3

from typing import Dict, Set, Tuple
import numpy as np
from loguru import logger

import megengine as mge
import megengine.module as M

__all__ = ["load_matched_weights", "unwarp_ckpt"]


def get_name_matched_keys(key: str, keys_to_match) -> Set:

    def match(key1: str, key2: str):
        """check if key1 match key2, key1 should be the shorter string"""
        return key1 == key2 or key2.endswith(key1)

    matched_keys = {k for k in keys_to_match if match(key, k)}
    return matched_keys


def get_shape_matched_keys(shape: tuple, keys_with_shape: Dict[str, Tuple]) -> Set:
    """
    NOTE: dumped shape and load shape of BatchNorm in MegEngine are different.
    What a stupid design !!!
    """
    return {k for k, v in keys_with_shape.items() if np.prod(v) == np.prod(shape)}


def unwarp_ckpt(weights, model_key="model") -> Dict:
    """unwarp checkpoint to get saved model state_dict"""
    if model_key in weights:
        weights = weights[model_key]
    if "state_dict" in weights:
        weights = weights["state_dict"]
    return weights


def full_match(weights, self_key_shape):
    """
    current matching logic:
        * first, matching name which totally-matched.
          e.g. "head.conv.weights" should match "head.conv.weights"
        * then, matching name which tail part matched.
          e.g. "conv.weights" should match "backbone.conv.weights"
        * Note that we assume only one name should matched.
          e.g. if "conv.weights" matches "head.conv.weights" and "backbone.conv.weights",
          if more than one keys are matched, shape will be used to filter keys.
          Only keys with the same shape will be matched, if there are still keys more than
          one to match, exception will be raised.
    """
    load_name_mapping = {}
    temp_unmatched_keys = {}
    unused_keys = []
    for w_key, w_value in weights.items():
        name_matched_keys = get_name_matched_keys(w_key, self_key_shape.keys())
        if not name_matched_keys:  # matched nothing
            unused_keys.append(w_key)
            continue

        if w_key in name_matched_keys:  # best match, highest priority
            match_key_name = w_key
        elif len(name_matched_keys) == 1:  # only match keys
            match_key_name = name_matched_keys.pop()
        else:
            shape_matched_keys = get_shape_matched_keys(
                w_value.shape, {k: self_key_shape[k] for k in name_matched_keys}
            )  # keys whose shape matched
            if len(shape_matched_keys) == 1:
                match_key_name = shape_matched_keys.pop()
            else:
                temp_unmatched_keys[w_key] = shape_matched_keys
                match_key_name = None

        if match_key_name is not None:
            # matched a key, filter previous multi-matched keys
            _filter_unmatched_keys(match_key_name, temp_unmatched_keys)
            self_key_shape.pop(match_key_name)  # pop matched_keys

        load_name_mapping[match_key_name] = w_key

    for temp_k, temp_v in temp_unmatched_keys.items():
        assert len(temp_v) == 1, f"{temp_k} matched more thant 1 keys: {list(temp_v)}"
        load_name_mapping[temp_v.pop()] = temp_k

    return load_name_mapping, unused_keys


def _filter_unmatched_keys(matched_key: str, unmatched_dict: Dict[str, Set]):
    for k, v in unmatched_dict.items():
        if matched_key in v:
            v.remove(matched_key)


def load_matched_weights(module: M.Module, weights: Dict, strict: bool = False) -> M.Module:
    """
    load state dict from weights for module. for matching logic, see meth:`match_mapping`.
    Note that strict=False has different meaning with strict=False of `module.load_state_dict`,
    If shape missmatched in this loading function, weights will not be loaded.

    Args:
        module: module to load state dict.
        weights: weights loading from, could be a string or BufferedReader.
        strict: strict matching or not.
    """
    if weights is None:
        return module

    if not isinstance(weights, dict):
        weights = unwarp_ckpt(mge.load(weights))

    self_key_shape = {k: v.shape for k, v in module.state_dict().items()}
    load_name_mapping, unused_keys = full_match(weights, self_key_shape)

    def load_func(k, v):
        # load_name_mapping is mapping from module.state_dict to weights
        if k not in load_name_mapping:  # weights not exist in load_name_mapping
            return v.numpy()

        load_key = load_name_mapping[k]
        value_to_load = weights[load_key]
        if value_to_load.shape != v.shape:
            if value_to_load.size == v.size:  # shape mismatched but size the same
                logger.warning(f"{k}{v.shape} loading from {load_key}{value_to_load.shape}")
                return value_to_load.reshape(v.shape)
            if not strict:
                logger.warning("Shape mismatched keys, skip loadding. Please double check.")
                logger.warning(f"Unmatch {k}{v.shape} from {load_key}{value_to_load.shape}")
                return v.numpy()
            else:
                raise ValueError(f"param `{k}` size mismatch, get {v.shape}")

        logger.info(f"{k} load from {load_key}, shape:{value_to_load.shape}")
        return value_to_load

    is_perfect_match = all([k == v for k, v in load_name_mapping.items()])
    if is_perfect_match:
        logger.info("Perfectly match weights during loading state dict...")
        # skip keys with unmatched shape
        if strict:
            module.load_state_dict(weights)
            return module
    else:
        logger.info("Smart load from weights...")

    module.load_state_dict(load_func)
    if unused_keys:
        logger.warning(f"keys not used in weights: {unused_keys}")
    return module
