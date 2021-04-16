import logging
import os
from datetime import datetime as dt

import zarr

from slaid.commons import Mask, BasicSlide
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.writers import ReducedSlide, _dump_masks, _get_slide_metadata

logger = logging.getLogger(__file__)


def dump(slide: BasicSlide,
         output_path: str,
         mask: str = None,
         overwrite: bool = False,
         **kwargs):
    logger.info('dumping slide to zarr on path %s', output_path)
    group = zarr.open_group(output_path)
    if not group.attrs:
        group.attrs.update(_get_slide_metadata(slide))
    _dump_masks(output_path, slide, overwrite, 'to_zarr', mask, **kwargs)


def load(path: str) -> BasicSlide:
    logger.info('loading slide from zarr at path %s', path)
    group = zarr.open_group(path)
    try:
        slide = EcvlSlide(group.attrs['filename'])
    except BasicSlide.InvalidFile:
        slide = ReducedSlide(group.attrs['filename'])
    for name, value in group.arrays():
        try:
            logger.info('loading mask %s, %s', name, value.attrs.asdict())
            kwargs = value.attrs.asdict()
            if 'datetime' in kwargs:
                kwargs['datetime'] = dt.fromtimestamp(kwargs['datetime'])
            slide.masks[name] = Mask(value, **kwargs)
        except Exception as ex:
            logger.error('skipping mask %s, exception: %s ', name, ex)
    return slide


def mask_exists(path: str, mask: 'str') -> bool:
    if not os.path.exists(path):
        return False
    group = zarr.open_group(path)
    return mask in group.array_keys()
