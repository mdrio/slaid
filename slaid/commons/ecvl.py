import logging
from typing import List, Tuple

import numpy as np
import pyecvl.ecvl as ecvl
from openslide import open_slide
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideGetLevels, OpenSlideRead
from pyeddl.tensor import Tensor as EddlTensor

from slaid.commons import Image as BaseImage
from slaid.commons import Slide as BaseSlide
from slaid.commons import Tensor as BaseTensor

logger = logging.getLogger('ecvl')


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
        else:
            array = array.transpose(0, 2, 1)
        return array

    def to_tensor(self):
        return ecvl.ImageToTensor(self._image)


class Slide(BaseSlide):
    def __init__(self, filename: str):
        self._level_dimensions = OpenSlideGetLevels(filename)
        if not self._level_dimensions:
            raise BaseSlide.InvalidFile(
                f'Cannot open file {filename}, is it a slide image?')
        super().__init__(filename)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._level_dimensions[0])

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


def load(filename: str):
    slide = Slide(filename)
    return slide
