import logging
from typing import List, Tuple

import numpy as np
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideImage

import slaid.commons.base as base
from slaid.commons.base import Image as BaseImage
from slaid.commons.base import ImageInfo

logger = logging.getLogger('ecvl')


class Image(BaseImage):
    IMAGE_INFO = ImageInfo.create('rgb', 'yx', 'first')

    def __init__(self, image: EcvlImage):
        self._image = image

    def to_array(self, image_info: ImageInfo = None):
        # FIXME
        array = np.array(self._image)
        array = array.transpose(0, 2, 1)

        if image_info is not None:
            array = self.IMAGE_INFO.convert(array, image_info)
        return array

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._image.dims_


class BasicSlide(base.BasicSlide):
    IMAGE_INFO = Image.IMAGE_INFO

    def __init__(self, filename: str):
        super().__init__(filename)
        self._slide = OpenSlideImage(filename)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._slide.GetLevelsDimensions()[0])

    def read_region(self, location: Tuple[int, int], level,
                    size: Tuple[int, int]) -> Image:
        return Image(self._slide.ReadRegion(level, location + size))

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.GetBestLevelForDownsample(downsample)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return [tuple(d) for d in self._slide.GetLevelsDimensions()]

    @property
    def level_downsamples(self):
        return self._slide.GetLevelDownsamples()


def load(filename: str):
    slide = BasicSlide(filename)
    return slide
