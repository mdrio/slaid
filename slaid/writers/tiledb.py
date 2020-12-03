import json
import logging
import os

import tiledb

from slaid.commons import Mask, Slide
from slaid.commons.ecvl import Slide as EcvlSlide
from slaid.writers import _dump_masks, _get_slide_metadata

logger = logging.getLogger(__file__)

SLIDE_INFO_FILENAME = '.slide_info'


def dump(slide: Slide,
         path: str,
         overwrite: bool = False,
         mask: str = False,
         ctx: tiledb.Ctx = None) -> str:
    if not os.path.isdir(path):
        logger.info('creating tiledb group at path %s', path)
        tiledb.group_create(path, ctx=ctx)
    output_path = os.path.join(path,
                               f'{os.path.basename(slide.filename)}.tiledb')
    os.makedirs(output_path, exist_ok=True)
    slide_filename = os.path.join(output_path, SLIDE_INFO_FILENAME)

    with open(slide_filename, 'w') as f:
        logger.debug('writing slide file on %s', slide_filename)
        json.dump(_get_slide_metadata(slide), f)

    _dump_masks(output_path, slide, overwrite, 'to_tiledb', mask, ctx=ctx)
    return output_path


def load(path: str, ctx: tiledb.Ctx = None) -> Slide:
    with open(os.path.join(path, SLIDE_INFO_FILENAME), 'r') as f:
        slide_info = json.load(f)
    slide = EcvlSlide(slide_info['filename'])
    masks = []
    tiledb.ls(path, lambda obj_path, obj_type: masks.append(obj_path), ctx=ctx)
    for name in masks:
        slide.masks[os.path.basename(name)] = Mask.from_tiledb(name, ctx=ctx)
    return slide
