from typing import List

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from slaid.models import Model as BaseModel


class Model(BaseModel):
    def __init__(self, filename, gpu=True):
        # Load the ANN tissue detector model implemented by using pyeddl
        # Create ANN topology (Same used during training phase)

        def tissue_detector_DNN():
            in_ = eddl.Input([3])

            layer = in_
            layer = eddl.ReLu(eddl.Dense(layer, 50))
            layer = eddl.ReLu(eddl.Dense(layer, 50))
            layer = eddl.ReLu(eddl.Dense(layer, 50))
            out = eddl.Softmax(eddl.Dense(layer, 2))
            net = eddl.Model([in_], [out])
            return net

        net = tissue_detector_DNN()
        eddl.build(net, eddl.rmsprop(0.00001), ["soft_cross_entropy"],
                   ["categorical_accuracy"],
                   eddl.CS_GPU() if gpu else eddl.CS_CPU())
        # Load weights
        eddl.load(net, filename, "bin")
        self._model = net

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        tensor = Tensor.fromarray(array)
        return eddl.predict(self._model, [tensor])

    def predict(self, array: np.ndarray) -> np.ndarray:
        #  np_img = np_img.transpose((1,2,0)) # Convert to channel last

        #  n_px = s[0] * s[1]
        #  array = array[:, :, :3].reshape(n_px, 3)

        predictions = self._predict(array)
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, 1])

        flat_mask = np.vstack(temp_mask)
        #  return flat_mask.reshape((s[0], s[1]))
        return flat_mask
