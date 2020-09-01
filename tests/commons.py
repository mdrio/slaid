from typing import List, Tuple

import numpy as np
from pyeddl.tensor import Tensor

from slaid.classifiers.eddl import Model as EddlModel
from slaid.commons import PandasPatchCollection, Slide
from slaid.commons.ecvl import Image


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
    class DummyIndexable:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, k):
            return self.value

    def __init__(
        self,
        ID: str,
        size: Tuple[int, int],
        patch_size: Tuple[int, int] = None,
        best_level_for_downsample: int = 1,
        level_downsample: int = 1,
        data=None,
    ):

        self._id = self._filename = ID
        self.size = size
        self.best_level_for_downsample = best_level_for_downsample
        self._level_dimensions = DummySlide.DummyIndexable(size)
        self._level_downsample = DummySlide.DummyIndexable(level_downsample)
        self.data = data
        self.features = {}
        self.masks = {}
        self.patch_size = patch_size if patch_size else size
        self.patches = PandasPatchCollection(self, patch_size,
                                             self.extraction_level)

    def __getstate__(self):
        return {'ID': self._filename, 'size': self.size}

    def __setstate__(self, dct):
        self.__init__(**dct)

    def __eq__(self, other):
        return self.ID == other.ID

    @property
    def dimensions(self):
        return self.size

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]):
        if self.data is None:
            return Image.new('RGB', size)
        else:
            data = self.data[location[1]:location[1] + size[1],
                             location[0]:location[0] + size[0]]
            mask = Image.fromarray(data, 'RGB')
            return mask

    @property
    def extraction_level(self):
        return 0

    def dimensions_at_level(self, level):
        return self.dimensions

    @property
    def dimensions_at_extraction_level(self):
        return self.dimensions

    @property
    def ID(self):
        return self._id

    def get_best_level_for_downsample(self, downsample: int):
        return self.best_level_for_downsample

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsample

    def __len__(self):
        return self.size[0] * self.size[1] // (self.patch_size[0] *
                                               self.patch_size[1])


class AllOneModel:
    def predict(self, array):
        return np.ones(array.shape[0], dtype=np.uint8)
