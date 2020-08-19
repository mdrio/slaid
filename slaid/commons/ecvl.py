import os
from typing import Dict, Tuple

from openslide import open_slide
from pyecvl.ecvl import OpenSlideGetLevels, OpenSlideRead

from slaid.commons import PATCH_SIZE, PatchCollection
from slaid.commons import Slide as BaseSlide


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

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]):
        return OpenSlideRead(self._filename, self._extraction_level,
                             location + size)

    def get_best_level_for_downsample(self, downsample: int):
        return open_slide(
            self._filename).get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return open_slide(self._filename).level_downsamples
