# NAPARI LAZY OPENSLIDE
#  Copyright (c) 2020, Trevor Manz
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of napari-lazy-openslide nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Tuple

import numpy as np
import zarr
from napari_lazy_openslide import OpenSlideStore
from openslide import open_slide
from PIL import Image as PIL_Image

from slaid.commons.base import Image as BaseImage, ImageInfo
from slaid.commons.base import Slide as BaseSlide


class Image(BaseImage):
    def __init__(self, image: PIL_Image):
        self._image = image

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._image.size)

    def to_array(self, image_info: ImageInfo) -> np.ndarray:
        array = np.array(self._image)  # yxc, RGB
        array = array[:, :, :3]
        if image_info.color_type == ImageInfo.COLORTYPE.BGR:
            array = array[..., ::-1]
        if image_info.coord == ImageInfo.COORD.XY:
            array = np.transpose(array, [1, 0, 2])
        if image_info.channel == ImageInfo.CHANNEL.FIRST:
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
