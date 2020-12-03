import logging
import os

import zarr

from slaid.commons import Mask, Slide
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.writers import ReducedSlide, _dump_masks, _get_slide_metadata

logger = logging.getLogger(__file__)


def dump(slide: Slide,
         output_path: str,
         mask: str = None,
         overwrite: bool = False,
         **kwargs):
    logger.info('dumping slide to zarr on path %s', output_path)
    group = zarr.open_group(output_path)
    if not group.attrs:
        group.attrs.update(_get_slide_metadata(slide))
    _dump_masks(output_path, slide, overwrite, 'to_zarr', mask, **kwargs)


def load(path: str) -> Slide:
    logger.info('loading slide from zarr at path %s', path)
    group = zarr.open_group(path)
    try:
        slide = EcvlSlide(group.attrs['filename'])
    except Slide.InvalidFile:
        slide = ReducedSlide(group.attrs['filename'])
    for name, value in group.arrays():
        logger.info('loading mask %s', name)
        slide.masks[name] = Mask(value, **value.attrs)
    return slide


def mask_exists(path: str, mask: 'str') -> bool:
    if not os.path.exists(path):
        return False
    group = zarr.open_group(path)
    return mask in group.array_keys()
