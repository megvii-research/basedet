#!/usr/bin/env python3
import contextlib

_FEATURES_DICT = {}
_HOOKS_DICT = {}


__all__ = [
    "ModuleInspector",
    "FeatureStorage",
    "feat_storage_to_dict",
    "feature_extract",
    "get_extracted_features_by_key",
    "release_extracted_features_by_key",
    "release_module_inspector_by_key",
]


class ModuleInspector:
    """
    Inspector to get module level feature/info
    :meth:`pre_forward_hook_func` is used to inspect feature/info before forward apply.
    :meth:`forward_hook_func` is used to inspect feature/info after forward apply.

    NOTE:
        1. no register backward hook in megengine.
    """

    def __init__(self, name):
        self.name = name
        self.value = None

    def pre_forward_hook_func(self, module, inpust):
        raise NotImplementedError

    def forward_hook_func(self, module, inputs, outputs):
        raise NotImplementedError


class FeatureStorage(ModuleInspector):
    """class used to storage feature after forward."""

    def hook_func(self, module, inputs, outputs):
        self.value = outputs


def feat_storage_to_dict(feat_dict):
    return {k: v.value for k, v in feat_dict.items()}


@contextlib.contextmanager
def feature_extract(module, names):
    """
    Build context to extract features from module using given names.

    .. code-block:: python

        with feature_extract(model, ["layer3", "layer4.conv1"]) as storage:
            model(inputs)
            feat_dict = feat_storage_to_dict(storage)

    Args:
        module (Module): megengine module.
        names (List[str]): module name used to extract features, e.g. "layer4" means
            feature after module.layer4 will be extracted. If no such module is found,
            exception will be raised.
    """
    if isinstance(names, str):
        names = [names]
    feat_dict = {}
    hooks = []
    for module_name, child_module in module.named_modules():
        # print(module_name)
        if module_name in names:
            feat_storage = FeatureStorage(module_name)
            feat_dict[module_name] = feat_storage
            hooks.append(child_module.register_forward_hook(feat_storage.hook_func))

    assert len(feat_dict) == len(names), "some names in {} are not found in module".format(names)
    yield feat_dict

    for h in hooks:
        h.remove()


def extract_module_feature(module, names, store_key=None):
    """extract features with given names and store them in dict using alias.

    Args:
        module (Module): module to extract features.
        names (Iterable[str]): feature names.
        store_key (Any): key to access storage dict. using module as key by default.
    """
    if store_key is None:
        store_key = module

    if isinstance(names, str):
        names = [names]

    hooks = []
    for module_name, child_module in module.named_modules():
        # print(module_name)
        if module_name in names:
            feat_storage = FeatureStorage(module_name)
            _FEATURES_DICT[store_key] = feat_storage
    _HOOKS_DICT[store_key] = hooks


def get_extracted_features_by_key(store_key):
    feat_dict = feat_storage_to_dict(_FEATURES_DICT[store_key])
    return feat_dict


def release_extracted_features_by_key(store_key):
    for key in store_key:
        if key in _FEATURES_DICT:
            _FEATURES_DICT[store_key].value = None


def release_module_inspector_by_key(store_key):
    _FEATURES_DICT.pop(store_key)
    for hook in _HOOKS_DICT[store_key]:
        hook.remove()
