from typing import List, Tuple

import numpy as np
from pyeddl.tensor import Tensor
from slaid.commons import Slide
from slaid.models.eddl import Model as EddlModel
from slaid.commons.base import Image


class GreenIsTissueModel:
    def predict(self, array: np.array) -> np.array:
        return array[:, 1] / 255


class EddlGreenIsTissueModel(EddlModel):
    def __init__(self):
        pass

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        array = array[:, 1] / 255
        array = np.array([(0, el) for el in array])
        tensor = Tensor.fromarray(array)
        return [tensor]


class DummyModel:
    def __init__(self, func):
        self.func = func

    def predict(self, array):
        return self.func(array.shape[0])


class DummySlide(Slide):
    def __init__(self,
                 level_dimensions: List[Tuple[int, int]],
                 level_downsamples: List[Tuple[int, int]],
                 ID=None):
        self._level_dimensions = level_dimensions
        self._level_downsamples = level_downsamples
        self._ID = ID

    @property
    def ID(self):
        return self._ID

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsamples

    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        raise NotImplementedError()

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError()
