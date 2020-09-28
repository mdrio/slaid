import os
from typing import List, Tuple

import numpy as np
import pyecvl.ecvl as ecvl
from openslide import open_slide
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideGetLevels, OpenSlideRead
from pyeddl.tensor import Tensor as EddlTensor

from slaid.commons import PATCH_SIZE
from slaid.commons import Image as BaseImage
from slaid.commons import Slide as BaseSlide
from slaid.commons import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __init__(self, tensor: EddlTensor):
        self._tensor = tensor

    def getdata(self):
        return self._tensor.getdata()


class Image(BaseImage):
    def __init__(self, image: EcvlImage):
        self._image = image

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._image.dims_)

    def to_array(self, PIL_FORMAT: bool = False) -> np.ndarray:
        array = np.array(self._image)
        if PIL_FORMAT:
            # convert to channel last
            array = array.transpose(2, 1, 0)
            # convert to rgb
            array = array[:, :, ::-1]
            #  array = np.flip(array, 1)
            #  array[:, :] = np.flip(array[:, :])
        return array

    def to_tensor(self):
        return ecvl.ImageToTensor(self._image)


class Slide(BaseSlide):
    def __init__(self, filename: str):
        if not os.path.exists(filename) or not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        self._level_dimensions = OpenSlideGetLevels(filename)
        super().__init__(filename)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._level_dimensions[0])

    @property
    def ID(self):
        return os.path.basename(self._filename)

    def read_region(self, location: Tuple[int, int], level,
                    size: Tuple[int, int]) -> Image:
        return Image(OpenSlideRead(self._filename, level, location + size))

    def get_best_level_for_downsample(self, downsample: int):
        return open_slide(
            self._filename).get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return [tuple(d) for d in self._level_dimensions]

    @property
    def level_downsamples(self):
        return open_slide(self._filename).level_downsamples


def create_slide(filename: str):
    slide = Slide(filename)
    return slide
