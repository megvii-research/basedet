#!/usr/bin/env python3

from typing import Dict

import megengine as mge
import megengine.module as M
from megengine import Tensor, jit

from basedet.utils import MeterBuffer, load_matched_weights


class BaseNet(M.Module):
    """Basic class for any network.

    Attributes:
        cfg (dict): a dict contains all settings for the network
    """

    def __init__(self):
        super().__init__()
        # extra_meter is used for logging extra used meter, such as accuracy.
        # user could use self.extra_meter.update(dict) to logging more info in basedet.
        self.extra_meter = MeterBuffer()

    def pre_process(self, inputs):
        """
        preprocess image for network. This function will convert inputs to Tensor
        and normalize it if img_mean and img_std are provided by subclasss.

        Args:
            inputs: input data to network.
        """
        if not isinstance(inputs, mge.Tensor):
            inputs = mge.Tensor(inputs)
        if hasattr(self, "img_mean") and self.img_mean is not None:
            inputs = inputs - self.img_mean
        if hasattr(self, "img_std") and self.img_std is not None:
            inputs = inputs / self.img_std
        return inputs

    def post_process(self, outputs):
        return outputs

    def network_forward(self, inputs):
        """
        pure network forward logic
        """
        pass

    def forward(self, inputs):
        if self.training:
            return self.get_losses(inputs)
        else:
            return self.inference(inputs)

    def get_losses(self, inputs) -> Dict[str, Tensor]:
        """ create(if have not create before) and return a dict which includes
        the whole losses within network.

        .. note::
            1. It must contains the `total` which indicates the total_loss in
               the returned loss dictionaries.
            2. Returned loss type must be OrderedDict.

        Args:
            inputs (dict[str, Tensor])

        Returns:
            loss_dict (OrderedDict[str, Tensor]): the OrderedDict contains the losses
        """
        pass

    def inference(self, inputs) -> Dict[str, Tensor]:
        """Run inference for network

        Args:
            inputs (dict[str, Tensor])
        """
        self.pre_process(inputs)
        outputs = self.network_forward(inputs)
        return self.post_process(outputs)

    def load_weights(self, weight_path: str, strict: bool = False) -> M.Module:
        """set weights of the network with the weight_path

        Args:
            weight_path (str): a file path of the weights
        """
        return load_matched_weights(self, weight_path, strict)

    def dump_weights(self, dump_path):
        mge.save({"state_dict": self.state_dict()}, dump_path)

    def dump_static_graph(self, inputs, optimize=True, graph_name="model.mge", **kwargs):
        from basedet.layers import adjust_stats
        with adjust_stats(self, training=False) as model:
            if isinstance(inputs, dict):
                inputs = inputs["data"]
            if not isinstance(inputs, mge.Tensor):
                inputs = mge.Tensor(inputs)

            @jit.trace(capture_as_const=True)
            def pred_func(data):
                outputs = model.network_forward(data)
                # TODO jit not support dict ouput now, change next a few lines in the future.
                if isinstance(outputs, dict):
                    outputs = [v for v in outputs.values()]
                return outputs

            # applied a forward pass to trace network.
            pred_func(inputs)
            if not optimize:
                graph_name = "unoptimized_" + graph_name
            pred_func.dump(graph_name, optimize_for_inference=optimize)

    def export(self):
        pass
