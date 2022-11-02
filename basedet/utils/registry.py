#!/usr/bin/env python3

import inspect

from basecore.utils import Registry


def is_type(obj, parent_type):
    return inspect.isclass(obj) and issubclass(obj, parent_type)


class registers:
    """All registried module could be found here."""
    trainers = Registry("trainers")
    hooks = Registry("hooks")
    dataloader = Registry("dataloader")
    models = Registry("models")
    solvers = Registry("solvers")
    evalutors = Registry("evalutors")
    losses = Registry("losses")

    # data related registry
    datasets = Registry("datasets")
    datasets_info = Registry("datasets info")
    transforms = Registry("transforms")
    schedulers = Registry("schedulers")


def register_mge_transform():
    import megengine.data.transform as T
    transforms = registers.transforms
    for name, obj in vars(T).items():
        if is_type(obj, T.Transform):
            transforms.register(obj, name="MGE_" + name)


def register_mge_dataset():
    import megengine.data.dataset as D
    datasets = registers.datasets
    for name, obj in vars(D).items():
        if is_type(obj, D.VisionDataset):
            datasets.register(obj, name=name)


def register_schedulers():
    import basecore.engine.lr_scheduler as S
    schedulers = registers.schedulers
    for name, obj in vars(S).items():
        if is_type(obj, S.LRScheduler):
            schedulers.register(obj, name=name)


def all_register():
    # try logic is used to avoid AssertionError of register twice
    try:
        # register hooks and trainers
        import basedet.engine
        # register models
        import basedet.models.det
        import basedet.models.cls
        # register solvers
        import basedet.solver
        # register dataloader
        import basedet.data  # noqa
        # register evaluators
        import basedet.evaluators # noqa
        # register dataset info
        import basedet.data.datasets  # noqa
        # register transforms
        import basedet.data.transforms # noqa
        register_mge_transform()
        register_mge_dataset()
        register_schedulers()
    except AssertionError:
        pass
