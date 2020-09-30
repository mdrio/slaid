import abc
import inspect
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

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


@dataclass
class Patch:
    x: int
    y: int
    size: Tuple[int, int]
    mask: "Mask" = None

    @property
    def area(self):
        return self.size[0] * self.size[1]


@dataclass
class Mask:
    array: np.array
    extraction_level: int
    level_downsample: int

    def ratio(self, patch: Patch) -> float:
        return np.sum(self.array[patch.y:patch.y + patch.size[1],
                                 patch.x:patch.x + patch.size]) / patch.area


class Slide(abc.ABC):
    def __init__(self, filename: str):
        self._filename = filename
        self.masks: Dict[str, Mask] = {}

    @abc.abstractproperty
    def dimensions(self) -> Tuple[int, int]:
        pass

    @abc.abstractproperty
    def ID(self):
        pass

    @abc.abstractmethod
    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        pass

    def patches(self, level: int, patch_size: Tuple[int, int]) -> Patch:
        dimensions = self.level_dimensions[level]
        for y in range(0, dimensions[1], patch_size[1]):
            for x in range(0, dimensions[0], patch_size[0]):
                location = (x, y)
                size = tuple([
                    min(patch_size[i], dimensions[i] - location[i])
                    for i in range(2)
                ])
                yield Patch(x, y, size)

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


def convert_patch(patch: Patch, slide: Slide, dest_mask: Mask) -> Patch:
    origin_mask = patch.mask
    origin_downsample = slide.level_downsamples[origin_mask.extraction_level]
    dest_downsample = slide.level_downsamples[dest_mask.extraction_level]
    print('origin_downsample', origin_downsample)
    print('dest_downsample', dest_downsample)
    factor = origin_downsample / dest_downsample
    size = (int(patch.size[0] * factor), int(patch.size[1] * factor))
    return Patch(int(patch.x * factor), int(patch.y * factor), size, dest_mask)
