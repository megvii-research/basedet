#!/usr/bin/env python3

import numpy as np

__all__ = ["DummyLoader"]


class DummyLoader:

    def __init__(self, batch_size=2, output_size=(800, 1344)):
        # output shape: (height, width)
        self.batch_size = batch_size
        self.output_size = output_size
        # dummy anno
        self.anno = np.array([[
            [  0.      ,   0.      , 800.      , 800.      ,  61.      ],  # noqa
            [148.33984 , 488.73206 , 667.7124  , 602.64056 ,  52.      ],  # noqa
            [170.45752 , 422.78433 , 572.1176  , 552.15686 ,  52.      ],  # noqa
            [228.24835 , 486.88892 , 600.71893 , 589.39874 ,  52.      ],  # noqa
            [ 71.803894,  54.444447, 110.20911 ,  78.19608 ,  43.      ],  # noqa
            [237.46405 ,   0.      , 418.64053 ,  32.03922 ,  41.      ],  # noqa
            [315.08798 , 101.472   , 464.52798 , 797.696   ,  80.      ],  # noqa
            [280.448   , 118.096   , 370.336   , 786.864   ,  70.      ],  # noqa
            [228.31999 , 104.71999 , 307.40802 , 791.456   ,  40.      ],  # noqa
            [145.61601 ,  94.736   , 246.288   , 786.86395 ,  20.      ],  # noqa
        ],
            [
            [315.08798 , 101.472   , 464.52798 , 797.696   ,  30.      ],  # noqa
            [280.448   , 118.096   , 370.336   , 786.864   ,  20.      ],  # noqa
            [228.31999 , 104.71999 , 307.40802 , 791.456   ,  10.      ],  # noqa
            [145.61601 ,  94.736   , 246.288   , 786.86395 ,  60.      ],  # noqa
            [ 68.32    , 101.12    , 244.496   , 787.872   ,  70.      ],  # noqa
            [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ],  # noqa
            [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ],  # noqa
            [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ],  # noqa
            [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ],  # noqa
            [  0.      ,   0.      ,   0.      ,   0.      ,   0.      ],  # noqa
        ],
        ], dtype="float32")
        scale_size = min(output_size[0] / 800, output_size[1] / 800)
        self.anno *= scale_size
        self.im_info = np.array([
            [*output_size, 612., 612., 10.],
            [*output_size, 500., 375., 5.]
        ], dtype="float32")

    def __iter__(self):
        return self

    def __next__(self):
        repeat = self.batch_size / len(self.anno)
        remain = self.batch_size % len(self.anno)

        def f(x, repeat, remain):
            repeat_x = np.repeat(x, repeat, axis=0)
            remain_x = x[:remain, ...]
            return np.concatenate([repeat_x, remain_x], axis=0)

        return {
            "data": np.random.random(size=(self.batch_size, 3, *self.output_size)),
            "gt_boxes": f(self.anno, repeat, remain),
            "im_info": f(self.im_info, repeat, remain),
        }
