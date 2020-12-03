import os
from slaid.commons.base import Slide


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
