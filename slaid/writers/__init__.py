import os
from typing import Dict, Tuple
from slaid.commons.base import Mask, Image, Slide


def _get_slide_metadata(slide: Slide) -> dict:
    return {'filename': slide.filename, 'resolution': slide.dimensions}


def _dump_masks(path: str,
                slide: Slide,
                overwrite: bool,
                func: str,
                only_mask: str = None,
                **kwargs):
    masks = {only_mask: slide.masks[only_mask]} if only_mask else slide.masks
    for name, mask_ in masks.items():
        getattr(mask_, func)(os.path.join(path, name), overwrite, **kwargs)


class ReducedSlide(Slide):
    def __init__(self, filename: str):
        self._filename = os.path.abspath(filename)
        self.masks: Dict[str, Mask] = {}

    def __eq__(self, other):
        return self._filename == other.filename and self.masks == other.masks

    @property
    def dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def filename(self):
        return self._filename

    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        raise NotImplementedError

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError

    @property
    def level_dimensions(self):
        raise NotImplementedError

    @property
    def level_downsamples(self):
        raise NotImplementedError
