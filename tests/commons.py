from typing import List, Tuple

import numpy as np
from pyeddl.tensor import Tensor

from slaid.commons.base import Slide
from slaid.commons.base import Image, ImageInfo
from slaid.models.eddl import Model as EddlModel


class BaseModel:
    image_info = ImageInfo.create('bgr', 'yx', 'first')

    def __init__(self, patch_size=None):
        self._patch_size = patch_size

    def __str__(self):
        return self.__class__.__name__

    @property
    def patch_size(self):
        return self._patch_size


class GreenModel(BaseModel):
    image_info = ImageInfo.create('rgb', 'yx', 'last')

    def predict(self, array: np.array) -> np.array:
        return array[:, 1] / 255


class EddlGreenModel(BaseModel, EddlModel):

    def __init__(self, patch_size=None):
        self.patch_size = patch_size

    @staticmethod
    def _create_net():
        pass

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        array = array[:, 1] / 255
        array = np.array([(0, el) for el in array])
        tensor = Tensor.fromarray(array)
        return [tensor]


class EddlGreenPatchModel(BaseModel, EddlModel):

    def __init__(self, patch_size=(256, 256)):
        self._patch_size = patch_size

    @staticmethod
    def _create_net():
        pass

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        prob_green = [
            np.sum(p[1, :, :] / 255) / (p.shape[1] * p.shape[2]) for p in array
        ]
        array = np.array([(0, prob) for prob in prob_green])
        tensor = Tensor.fromarray(array)
        return [tensor]


class BaseDummyModel(BaseModel):

    def __init__(self, patch_size=None):
        self._patch_size = patch_size
        self.array_predicted = []

    def predict(self, array):
        self.array_predicted.append(array)
        return self._predict(array)

    def _predict(self, array):
        raise NotImplementedError

    @property
    def patch_size(self):
        return self._patch_size


class DummyModel(BaseDummyModel):
    image_info = ImageInfo.create('bgr', 'yx', 'first')

    def __init__(self, func, patch_size=None):
        self.func = func
        super().__init__(patch_size)

    def _predict(self, array):
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

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> Image:
        raise NotImplementedError()

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError()


class AllOneModel(BaseDummyModel):

    def _predict(self, array):
        return np.ones(array.shape[0], dtype=np.uint8)

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError()
