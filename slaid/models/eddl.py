import logging
import os
import sys
import tarfile
import tempfile
from typing import List

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from abc import ABC, abstractstaticmethod
from slaid.models import Model as BaseModel

logger = logging.getLogger('eddl-models')


class Model(BaseModel, ABC):
    patch_size = None
    channel_first = False

    def __init__(self, weight_filename, gpu=True):
        self._weight_filename = weight_filename
        self._gpu = gpu
        self._create_model()

    def _create_model(self):
        net = self._create_net()
        eddl.build(net, eddl.rmsprop(0.00001), ["soft_cross_entropy"],
                   ["categorical_accuracy"],
                   eddl.CS_GPU() if self._gpu else eddl.CS_CPU())
        eddl.load(net, self._weight_filename, "bin")
        self._model = net

    @abstractstaticmethod
    def _create_net():
        pass

    def predict(self, array: np.ndarray) -> np.ndarray:
        predictions = self._predict(array)
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, 1])

        flat_mask = np.vstack(temp_mask)
        return flat_mask

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        if self.channel_first:
            logger.debug('array.shape %s', array.shape)
            array = array.transpose(0, 3, 2, 1)
        tensor = Tensor.fromarray(array)
        #  if self.patch_size:
        #      tensor = Tensor.unsqueeze(tensor)
        return eddl.predict(self._model, [tensor])

    def __getstate__(self):
        return dict(weight_filename=self._weight_filename, gpu=self._gpu)

    def __setstate__(self, state):
        self.__init__(**state)


class TissueModel(Model):
    @staticmethod
    def _create_net():
        in_ = eddl.Input([3])
        layer = in_
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        out = eddl.Softmax(eddl.Dense(layer, 2))
        net = eddl.Model([in_], [out])
        return net


class TumorModel(Model):
    patch_size = (256, 256)
    channel_first = True

    @staticmethod
    def _create_net():
        in_size = [256, 256]
        num_classes = 2
        lr = 1e-5
        in_ = eddl.Input([3, in_size[0], in_size[1]])
        out = TumorModel._create_VGG16(in_, num_classes)
        net = eddl.Model([in_], [out])
        return net

    @staticmethod
    def _create_VGG16(in_layer, num_classes, seed=1234, init=eddl.HeNormal):
        x = in_layer
        x = eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.Reshape(x, [-1])
        x = eddl.ReLu(init(eddl.Dense(x, 256), seed))
        x = eddl.Softmax(eddl.Dense(x, num_classes))
        return x
