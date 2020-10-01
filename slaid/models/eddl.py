from typing import List

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from slaid.models import Model as BaseModel


class Model(BaseModel):
    def __init__(self, filename, gpu=True):
        # Load the ANN tissue detector model implemented by using pyeddl
        # Create ANN topology (Same used during training phase)

        self._filename = filename
        self._gpu = gpu
        net = self._create_model(gpu)
        eddl.build(net, eddl.rmsprop(0.00001), ["soft_cross_entropy"],
                   ["categorical_accuracy"],
                   eddl.CS_GPU() if gpu else eddl.CS_CPU())
        # Load weights
        eddl.load(net, filename, "bin")
        self._model = net

    @staticmethod
    def _create_model(gpu: bool):
        in_ = eddl.Input([3])

        layer = in_
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        out = eddl.Softmax(eddl.Dense(layer, 2))
        net = eddl.Model([in_], [out])
        return net

    def predict(self, array: np.ndarray) -> np.ndarray:
        predictions = self._predict(array)
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, 1])

        flat_mask = np.vstack(temp_mask)
        return flat_mask

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        tensor = Tensor.fromarray(array)
        return eddl.predict(self._model, [tensor])

    def __getstate__(self):
        return dict(filename=self._filename, gpu=self._gpu)

    def __setstate__(self, state):
        self.__init__(**state)
