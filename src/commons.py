from openslide import OpenSlide
from typing import Tuple, Dict
import os
import sys
import inspect


def get_class(name, module):
    return dict(inspect.getmembers(sys.modules[module], inspect.isclass))[name]


class Slide:
    def __init__(self, filename: str):
        self._filename = filename
        self._slide = OpenSlide(filename)

    @property
    def dimensions(self):
        return self._slide.dimensions

    @property
    def ID(self):
        return os.path.basename(self._filename)

    def read_region(self, location: Tuple[int, int], level: int,
                    size: Tuple[int, int]):
        return self._slide.read_region(location, level, size)

    def __getstate__(self):
        return self._filename

    def __setstate__(self, filename):
        self.__init__(filename)

    def iterate_by_patch(self, patch_size: Tuple[int, int] = None):
        patch_size = patch_size if patch_size else (256, 256)
        for y in range(0, self.dimensions[1], patch_size[1]):
            for x in range(0, self.dimensions[0], patch_size[0]):
                yield Patch(self, (x, y), patch_size)

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples


class SlideIterator:
    def __init__(self, slide: Slide, patch_size: Tuple[int, int]):
        self._slide = slide
        self._patch_size = patch_size

    def __iter__(self):
        return self._slide.iterate_by_patch(self._patch_size)


class Patch:
    def __init__(self,
                 slide: Slide,
                 coordinates: Tuple[int, int],
                 size: Tuple[int, int],
                 features: Dict = None):
        self.slide = slide
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.features = features or {}

    def __str__(self):
        return (f'slide: {self.slide}, x: {self.x}, '
                f'y: {self.y}, size: {self.size}')
