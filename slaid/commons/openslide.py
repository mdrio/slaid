from typing import Tuple

import numpy as np
import zarr
from napari_lazy_openslide import OpenSlideStore
from openslide import open_slide
from PIL import Image as PIL_Image

from slaid.commons.base import Image as BaseImage
from slaid.commons.base import Slide as BaseSlide


class Image(BaseImage):
    def __init__(self, image: PIL_Image):
        self._image = image

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._image.size)

    def to_array(self, colortype: "Image.COLORTYPE", coords: "Image.COORD",
                 channel: 'Image.CHANNEL') -> np.ndarray:
        array = np.array(self._image)  # yxc, RGB
        array = array[:, :, :3]
        if self.COLORTYPE(colortype) == self.COLORTYPE.BGR:
            array = array[..., ::-1]
        if self.COORD(coords) == self.COORD.XY:
            array = np.transpose(array, [1, 0, 2])
        if self.CHANNEL(channel) == self.CHANNEL.FIRST:
            array = np.transpose(array, [2, 0, 1])
        return array
        #  if PIL_FORMAT:
        #      # convert to channel last
        #      array = array.transpose(2, 1, 0)
        #      # convert to rgb
        #      array = array[:, :, ::-1]
        #  else:
        #      array = array.transpose(0, 2, 1)


class Slide(BaseSlide):
    def __init__(self, filename: str):
        super().__init__(filename)
        slide = open_slide(filename)  # not serializable...
        self._dimensions = slide.dimensions
        self._level_dimensions = slide.level_dimensions
        self._level_downsamples = slide.level_downsamples

    def __eq__(self, other):
        return self._filename == other.filename and self.masks == other.masks

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._dimensions

    @property
    def filename(self):
        return self._filename

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]) -> BaseImage:
        return Image(
            open_slide(self.filename).read_region(location, level, size))

    def get_best_level_for_downsample(self, downsample: int):
        return open_slide(
            self.filename).get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsamples

    def to_array(self, level):
        store = OpenSlideStore(self.filename)
        grp = zarr.open(store, mode="r")
        datasets = grp.attrs["multiscales"][0]["datasets"]

        pyramid = [grp.get(d["path"]) for d in datasets]
        return pyramid[level]


def load(filename: str):
    slide = Slide(filename)
    return slide
