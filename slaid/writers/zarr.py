import logging
import os

import zarr

from slaid.commons import Mask, Slide
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.writers import _get_slide_metadata
from slaid.writers import _dump_masks
logger = logging.getLogger(__file__)


def dump(slide: Slide,
         path: str,
         mask: str = None,
         overwrite: bool = False,
         **kwargs) -> str:
    output_path = os.path.join(path,
                               f'{os.path.basename(slide.filename)}.zarr')
    logger.info('dumping slide to zarr on path %s', output_path)
    group = zarr.open_group(output_path)
    if not group.attrs:
        group.attrs.update(_get_slide_metadata(slide))
    _dump_masks(output_path, slide, overwrite, 'to_zarr', mask, **kwargs)
    return output_path


def load(path: str) -> Slide:
    logger.info('loading slide from zarr at path %s', path)
    group = zarr.open_group(path)
    slide = EcvlSlide(group.attrs['filename'])
    for name, value in group.arrays():
        logger.info('loading mask %s', name)
        slide.masks[name] = Mask(value, **value.attrs)
    return slide
