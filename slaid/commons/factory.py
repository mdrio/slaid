import logging
import os
from importlib import import_module

from slaid.commons import BaseSlideFactory, SlideStore
from slaid.writers import REGISTRY

logger = logging.getLogger('slaid.commons.factory')


class SlideFactory(BaseSlideFactory):
    def get_slide(self):
        basic_slide_cls = import_module(
            f'slaid.commons.{self._basic_slide_module}').BasicSlide
        slide_cls = import_module(f'slaid.commons.{self._slide_module}').Slide

        slide_ext_with_dot = os.path.splitext(self._filename)[-1]
        slide_ext = slide_ext_with_dot[1:]
        try:
            basic_slide = REGISTRY[slide_ext].load(self._filename)
        except KeyError:
            basic_slide = basic_slide_cls(self._filename)
        store = SlideStore(basic_slide, self._tilesize)
        return slide_cls(store, self._image_info)
