from openslide import open_slide, ImageSlide
from typing import Tuple, Dict
import os

from slaid.commons.base import Slide as BaseSlide
from slaid.commons.base import PATCH_SIZE
from slaid.commons.base import PandasPatchCollection


class Slide(BaseSlide):
    def __init__(self,
                 filename: str,
                 features: Dict = None,
                 patches: 'PatchCollection' = None,
                 patch_size: Tuple[int, int] = PATCH_SIZE,
                 extraction_level=2):
        self._slide = open_slide(filename)
        super().__init__(filename, features, patches, patch_size,
                         extraction_level)

    #  def get_thumbnail(self) -> "Slide":
    #      _slide = ImageSlide(
    #          self._slide.get_thumbnail(self.dimensions_at_extraction_level))
    #
    #      slide = Slide(self._filename,
    #                    self.features,
    #                    self.patches,
    #                    extraction_level=self._extraction_level)
    #      slide._slide = _slide
    #      return slide
    #
    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._slide.dimensions)

    @property
    def dimensions_at_extraction_level(self) -> Tuple[int, int]:
        return tuple(self._slide.level_dimensions[self._extraction_level])

    @property
    def ID(self):
        return os.path.basename(self._filename)

    def read_region(self, location: Tuple[int, int], size: Tuple[int, int]):
        return self._slide.read_region(location, self._extraction_level, size)

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._slide.level_dimensions

    @property
    def level_downsamples(self):
        return self._slide.level_downsamples
