import os
from typing import Dict, Tuple

import numpy as np
from openslide import open_slide
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideGetLevels, OpenSlideRead

from slaid.commons import PATCH_SIZE
from slaid.commons import Image as BaseImage
from slaid.commons import PatchCollection
from slaid.commons import Slide as BaseSlide


class Image(BaseImage):
    def __init__(self, image: EcvlImage):
        self._image = image

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._image.dims_)

    def to_array(self, PIL_FORMAT: bool = False) -> np.ndarray:
        array = np.array(self._image)
        if PIL_FORMAT:
            array = array.transpose(1, 2, 0)
            array[:, :] = np.flip(array[:, :])
        return array


class Slide(BaseSlide):
    def __init__(self,
                 filename: str,
                 features: Dict = None,
                 patches: PatchCollection = None,
                 patch_size: Tuple[int, int] = PATCH_SIZE,
                 extraction_level=2):
        self._level_dimensions = OpenSlideGetLevels(filename)
        #  if not self._level_dimensions:
        #      self._level_dimensions = open_slide(filename).level_dimensions
        super().__init__(filename, features, patches, patch_size,
                         extraction_level)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._level_dimensions[0])

    @property
    def dimensions_at_extraction_level(self) -> Tuple[int, int]:
        return tuple(self._level_dimensions[self._extraction_level])

    @property
    def ID(self):
        return os.path.basename(self._filename)

    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        return Image(
            OpenSlideRead(self._filename, self._extraction_level,
                          location + size))

    def get_best_level_for_downsample(self, downsample: int):
        return open_slide(
            self._filename).get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return open_slide(self._filename).level_downsamples
