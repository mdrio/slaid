import abc
import inspect
import sys
from dataclasses import dataclass
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


@dataclass
class Patch:
    x: int
    y: int
    size: Tuple[int, int]
    level_downsample: float

    @property
    def area(self):
        return self.size[0] * self.size[1]


@dataclass
class Mask:
    array: np.ndarray
    extraction_level: int
    level_downsample: int

    def ratio(self, patch: Patch) -> float:
        area = self._convert_patch_to_area(patch)
        return np.sum(self.array[area[1]:area[1] + area[3], area[0]:area[0] +
                                 area[3]]) / (area[2] * area[3])

    def _convert_patch_to_area(self,
                               patch: Patch) -> Tuple[int, int, int, int]:
        return tuple(
            round(_ * patch.level_downsample / self.level_downsample)
            for _ in (patch.x, patch.y) + patch.size)

    def to_image(self):
        return PIL.Image.fromarray(255 * self.array, 'L')

    def show(self):
        self.to_image().show()

    def save(self, path):
        self.to_image().save(path)


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

    def patches(self,
                level: int,
                patch_size: Tuple[int, int],
                start: Tuple[int, int] = None,
                end: Tuple[int, int] = None) -> Patch:
        dimensions = self.dimensions
        downsample = self.level_downsamples[level]
        start = start or (0, 0)
        end = end or dimensions
        step = tuple(round(_ * round(downsample)) for _ in patch_size)
        for y in range(start[1], end[1], step[1]):
            for x in range(start[0], end[0], step[0]):
                location = (x, y)
                size = tuple(
                    min(patch_size[i],
                        round((dimensions[i] - location[i]) /
                              downsample), end[i]) for i in range(2))
                yield Patch(x, y, size, self.level_downsamples[level])

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


def convert_patch(patch: Patch, slide: Slide, dest_downsample: float) -> Patch:
    origin_downsample = patch.level_downsample
    factor = origin_downsample / dest_downsample
    size = (int(patch.size[0] * factor), int(patch.size[1] * factor))
    return Patch(patch.x, patch.y, size, dest_downsample)
