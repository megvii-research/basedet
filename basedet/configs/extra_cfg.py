#!/usr/bin/env python3

from basecore.config import ConfigDict

__all__ = [
    "DataConfig",
    "SolverConfig",
    "TrainerConfig",
    "TestConfig",
    "GlobalConfig",
]


class DataConfig(ConfigDict):

    def __init__(self):
        self.BUILDER_NAME = "DataloaderBuilder"
        self.TRAIN = dict(
            name="coco_2017_train",
            remove_images_without_annotations=True,
            order=("image", "boxes", "boxes_category", "info"),
        )
        self.TEST = dict(
            name="coco_2017_val",
            remove_images_without_annotations=False,
            order=("image", "info"),
        )
        self.NUM_CLASSES = 80
        self.NUM_WORKERS = 2
        self.ENABLE_INFINITE_SAMPLER = True


class GlobalConfig(ConfigDict):

    def __init__(self):
        self.OUTPUT_DIR = "logs"
        self.CKPT_SAVE_DIR = "/data/Outputs/model_logs/basedet_playground"
        # use the following ckpt_save_dir for oss user
        # CKPT_SAVE_DIR="s3://basedet/playground/" + self.user,
        self.LOG_INTERVAL = 20
        self.TENSORBOARD = dict(
            ENABLE=False,
        )


class ModelConfig(ConfigDict):

    def __init__(self):
        self.BATCHSIZE = 2
        self.WEIGHTS = None
        self.BACKBONE = dict(
            NAME="resnet50",
            IMG_MEAN=[103.530, 116.280, 123.675],  # BGR
            IMG_STD=[57.375, 57.12, 58.395],
            NORM="FrozenBN",
            FREEZE_AT=2,
        )


class SolverConfig(ConfigDict):

    def __init__(self):
        self.BUILDER_NAME = "DetSolver"
        self.OPTIMIZER_NAME = "SGD"
        self.LR_SCHEDULER_NAME = "MultiStepLR"
        self.BASIC_LR = 0.01 / 16.0  # The basic learning rate for single-image
        self.WEIGHT_DECAY = 1e-4
        self.EXTRA_OPT_ARGS = dict(
            momentum=0.9,
        )
        self.REDUCE_MODE = "MEAN"
        self.EPOCHWISE_STEP = False
        self.WARM_ITERS = 500
        self.NUM_IMAGE_PER_EPOCH = 80000
        self.MAX_EPOCH = 18
        self.LR_DECAY_STAGES = [12, 16]
        self.LR_DECAY_RATE = 0.1
        self.EXTRA_LR_ARGS = dict()


class TrainerConfig(ConfigDict):

    def __init__(self):
        self.NAME = "DetTrainer"
        self.RESUME = False
        self.AMP = dict(
            ENABLE=False,
            # when dynamic scale is enabled, we start with a higher scale of 65536,
            # scale is doubled every 2000 iter or halved once inf is detected during training.
            DYNAMIC_SCALE=False,
        )
        self.EMA = dict(
            ENABLE=False,
            ALPHA=5e-4,
            MOMENTUM=None,
            UPDATE_PERIOD=1,
            BURNIN_ITER=2000,
        )
        self.GRAD_CLIP = dict(
            ENABLE=False,
            # supported type: ("value", "norm")
            TYPE="value",
            ARGS=dict(lower=-1, upper=1)
            # ARGS=dict(max_norm=1.0, ord=2)
        )


class TestConfig(ConfigDict):

    def __init__(self):
        self.EVALUATOR_NAME = "COCOEvaluator"
        self.MAX_BOXES_PER_IMAGE = 100
        self.IMG_MIN_SIZE = 800
        self.IMG_MAX_SIZE = 1333
        self.VIS_THRESHOLD = 0.3
        self.CLS_THRESHOLD = 0.05
        self.IOU_THRESHOLD = 0.5
        self.EVAL_EPOCH_INTERVAL = None
        self.AUG = dict(
            VALUE=(
                (
                    "MGE_ShortestEdgeResize",
                    dict(
                        min_size=self.IMG_MIN_SIZE,
                        max_size=self.IMG_MAX_SIZE,
                        sample_style="choice"
                    ),
                ),
                ("ToMode", dict(mode="NCHW")),
            ),
            WRAPPER=(("TestTimeCompose", dict(),),),
        )
