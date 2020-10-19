import abc
import inspect
import os
import sys
from typing import Dict, Tuple

import numpy as np
import PIL

PATCH_SIZE = (256, 256)


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


class Tensor(abc.ABC):
    @abc.abstractmethod
    def getdata() -> np.ndarray:
        pass


class Image(abc.ABC):
    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def to_array(self, PIL_FORMAT: bool = False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def to_tensor(self):
        pass


class Mask:
    def __init__(self, array: np.ndarray, extraction_level: int,
                 level_downsample: int):
        self.array = array
        self.extraction_level = extraction_level
        self.level_downsample = level_downsample

    def to_image(self):
        return PIL.Image.fromarray(255 * self.array, 'L')


class Slide(abc.ABC):
    def __init__(self, filename: str):
        if not os.path.exists(filename) or not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        self._filename = os.path.abspath(filename)
        self.masks: Dict[str, Mask] = {}

    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @property
    def filename(self):
        return self._filename

    @abc.abstractmethod
    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        pass

    @abc.abstractmethod
    def get_best_level_for_downsample(self, downsample: int):
        pass

    @abc.abstractproperty
    def level_dimensions(self):
        pass

    @abc.abstractproperty
    def level_downsamples(self):
        pass


def round_to_patch(coordinates, patch_size):
    res = []
    for i, c in enumerate(coordinates):
        size = patch_size[i]
        q, r = divmod(c, size)
        res.append(size * (q + round(r / size)))
    return tuple(res)
