import pickle
from typing import List, Tuple

import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from slaid.classifiers import Model as BaseModel
from slaid.classifiers import TissueMaskPredictor as BaseTissueMaskPredictor
from slaid.commons.ecvl import Image


def load_model(model_weights, gpu=True):
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
    eddl.load(net, model_weights, "bin")

    return net


class Model(BaseModel):
    def __init__(self, model):
        self._model = model

    def predict(self, array: np.array) -> np.array:
        #  np_img = np_img.transpose((1,2,0)) # Convert to channel last

        s = array.shape
        n_px = s[0] * s[1]
        array = array[:, :, :3].reshape(n_px, 3)

        t_eval = Tensor.fromarray(array)

        predictions = eddl.predict(self._model,
                                   [t_eval])  # Prediction.. get probabilities
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, 1])

        flat_mask = np.vstack(temp_mask)
        return flat_mask.reshape((s[0], s[1]))


#  class TissueMaskPredictor(BaseTissueMaskPredictor):
#      @staticmethod
#      def create(model_filename):
#          with open(model_filename, 'rb') as f:
#              return TissueMaskPredictor(pickle.load(f))
#
#      def __init__(self, model):
#          self._model = model
#
#      def get_tissue_mask(self, image: Image, threshold: float) -> np.array:
#          array = image.to_array()
#          predictions = self._model.predict(image.to_array())
#
#          mask = self._get_binary_mask(
#              predictions, array.shape,
#              threshold)  # Get the actual mask (binarization)
#          return mask
#
#          print('mask shape', mask.shape)
#          return mask
#
#      def _get_binary_mask(self, prob_tensors: List[Tensor],
#                           shape: Tuple[int, int], th: float) -> np.ndarray:
#          """
#          @prob_T_L: list of probability tensors resulting
#              from model predictions.
#              Once each tensor is binarized, a single array is created stacking
#              them and the image mask is created reshaping the array
#          """
#
#          mask_np_l = []
#          for prob_T in prob_tensors:
#              output_np = prob_T.getdata()
#              pred_np = np.zeros(output_np.shape[0])
#              pred_np[output_np[:, 1] > th] = 1
#              mask_values = pred_np
#              mask_np_l.append(mask_values)
#
#          mask_values = np.vstack(mask_np_l)
#          mask = mask_values.reshape((shape[0], shape[1]))
#          return mask
