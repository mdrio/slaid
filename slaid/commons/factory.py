import logging
import os
from importlib import import_module

import slaid.writers.tiledb as tiledb_io
import slaid.writers.zarr as zarr_io
from slaid.commons import BaseSlideFactory, SlideStore

logger = logging.getLogger('slaid.commons.factory')


class SlideFactory(BaseSlideFactory):
    _STORAGE = {'zarr': zarr_io.load, 'tiledb': tiledb_io.load}

    def get_slide(self):
        basic_slide_cls = import_module(
            f'slaid.commons.{self._basic_slide_module}').BasicSlide
        slide_cls = import_module(f'slaid.commons.{self._slide_module}').Slide

        slide_ext_with_dot = os.path.splitext(self._filename)[-1]
        slide_ext = slide_ext_with_dot[1:]
        basic_slide = self._STORAGE.get(slide_ext,
                                        basic_slide_cls)(self._filename)
        store = SlideStore(basic_slide, self._tilesize)
        return slide_cls(store, self._image_info)
