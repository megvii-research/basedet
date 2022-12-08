#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. All rights reserved.

import megfile
from loguru import logger

import megengine.distributed as dist
from megengine.data import DataLoader
from megengine.module import Module

from basedet.utils import registers

from .base_cfg import BaseConfig
from .extra_cfg import (
    DataConfig,
    GlobalConfig,
    ModelConfig,
    SolverConfig,
    TestConfig,
    TrainerConfig
)


class DetectionConfig(BaseConfig):

    def __init__(self, cfg=None, **kwargs):
        """
        params in kwargs is the latest value
        """
        super().__init__(cfg, **kwargs)
        self.MODEL: ModelConfig = ModelConfig()
        self.DATA: DataConfig = DataConfig()
        self.SOLVER: SolverConfig = SolverConfig()

        # training
        self.TRAINER: TrainerConfig = TrainerConfig()
        self.HOOKS = dict(
            BUILDER_NAME="SimpleHookList",
        )
        # testing
        self.TEST: TestConfig = TestConfig()
        self.AUG = dict(
            TRAIN_VALUE=(
                (
                    "MGE_ShortestEdgeResize",
                    dict(min_size=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style="choice")  # noqa
                ),
                ("MGE_RandomHorizontalFlip", dict(prob=0.5)),
                ("MGE_ToMode", dict(mode="CHW")),
            ),
            TRAIN_WRAPPER=(("MGE_Compose", dict(order=("image", "boxes", "boxes_category"),)),),
        )
        self.GLOBAL: GlobalConfig = GlobalConfig()

    def build_model(self) -> Module:
        model = registers.models.get(self.MODEL.NAME)(self)
        return model

    def build_dataloader(self) -> DataLoader:
        dataloader = registers.dataloader.get(self.DATA.BUILDER_NAME).build(self)
        return dataloader

    def build_solver(self, model):
        solver = registers.solvers.get(self.SOLVER.BUILDER_NAME).build(self, model)
        return solver

    def build_trainer(self):
        logger.info("Using model named {}".format(self.MODEL.NAME))
        model = self.build_model()

        weights = self.MODEL.WEIGHTS
        if not weights:
            logger.warning("Train model from scrach...")
        else:
            logger.info("Loading model weights from {}".format(weights))
            with megfile.smart_open(weights, "rb") as f:
                model.load_weights(f)

        # sync parameters
        if dist.get_world_size() > 1:
            dist.bcast_list_(model.parameters(), dist.WORLD)
            dist.bcast_list_(model.buffers(), dist.WORLD)

        logger.info("Using dataloader named {}".format(self.DATA.BUILDER_NAME))
        dataloader = self.build_dataloader()

        solver_builder_name = self.SOLVER.BUILDER_NAME
        logger.info("Using solver named {}".format(solver_builder_name))
        solver = self.build_solver(model)

        hooks_builder_name = self.HOOKS.BUILDER_NAME
        logger.info("Using hook list named {}".format(hooks_builder_name))
        hookslist = self.build_hooks()

        trainer_name = self.TRAINER.NAME
        logger.info("Using trainer named {}".format(trainer_name))
        register_trainer = registers.trainers.get(trainer_name)
        trainer = register_trainer(self, model, dataloader, solver, hooks=hookslist)
        return trainer

    def build_evaluator(self):
        evaluator_name = self.TEST.EVALUATOR_NAME
        evaluator = registers.evalutors.get(evaluator_name)(self)
        return evaluator

    def build_hooks(self, hooks=None):
        if hooks is None:
            hooks = registers.hooks.get(self.HOOKS.BUILDER_NAME).build(self)
        return hooks
