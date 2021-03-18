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

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import zarr
from napari_lazy_openslide import OpenSlideStore
from napari_lazy_openslide.store import (ArgumentError, _parse_chunk_path,
                                         init_attrs)
from openslide import open_slide
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideGetLevels, OpenSlideRead
from zarr.storage import init_array, init_group

from slaid.commons import Image as BaseImage
from slaid.commons import ImageInfo
from slaid.commons import Slide as BaseSlide
from slaid.commons import SlideArray
from slaid.commons import SlideArrayFactory as BaseSlideArrayFactory

logger = logging.getLogger('ecvl')


class Image(BaseImage):
    def __init__(self, image: EcvlImage):
        self._image = image

    def __array__(self):
        return self._image.__array__()


def create_meta_store(slide: "Slide", tilesize: int) -> Dict[str, bytes]:
    """Creates a dict containing the zarr metadata for the multiscale openslide image."""
    store = dict()
    root_attrs = {
        "multiscales": [{
            "name":
            Path(slide._filename).name,
            "datasets": [{
                "path": str(i)
            } for i in range(slide.level_count)],
            "version":
            "0.1",
        }]
    }
    init_group(store)
    init_attrs(store, root_attrs)
    for i, (x, y) in enumerate(slide.level_dimensions):
        init_array(
            store,
            path=str(i),
            shape=(3, y, x),
            chunks=(3, tilesize, tilesize),
            dtype="|u1",
            compressor=None,
        )
    return store


class EcvlStore(OpenSlideStore):
    def __init__(self, slide: "Slide", tilesize: int = 512):
        self._path = slide.filename
        self._slide = slide
        self._tilesize = tilesize
        self._store = create_meta_store(self._slide, tilesize)
        #  self._image_info = image_info if image_info else ImageInfo()

    def __getitem__(self, key: str):
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
            tile = self._slide.read_region(location, level, size)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except:
            # TODO: probably need better error handling.
            # If anything goes wrong, we just signal the chunk
            # is missing from the store.
            raise KeyError(key)

        return tile.to_array(ImageInfo('bgr', 'yx', 'first'))
        #  return np.array(tile).tobytes()


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


class SlideArrayFactory(BaseSlideArrayFactory):
    IMAGE_INFO = ImageInfo('bgr', 'yx', 'first')

    def __init__(self, slide: Slide, image_info: ImageInfo):
        self._slide = slide
        self.image_info = image_info
        store = EcvlStore(slide)
        grp = zarr.open(store, mode="r")
        multiscales = grp.attrs["multiscales"][0]
        self._pyramid = [
            SlideArray(da.from_zarr(store, component=d["path"]),
                       self.IMAGE_INFO).convert(image_info)
            for d in multiscales["datasets"]
        ]

    def __getitem__(self, key) -> "SlideArray.Level":
        return self._pyramid[key]
