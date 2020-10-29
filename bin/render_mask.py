#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

from clize import run

from slaid.renderers import JSONEncoder, from_zarr


def main(zarr_archive,
         mask_name,
         output,
         *,
         downsample: ('d', int) = 1,
         threshold: (float, 't') = None):

    s = from_zarr(zarr_archive)
    mask = s.masks[mask_name]
    ext = os.path.splitext(output)[-1]

    if ext == '.json':
        pols = mask.to_polygons(0.8, downsample=downsample)
        json.dump(pols, open(output, 'w'), cls=JSONEncoder)
    else:
        mask.to_image(downsample, threshold).save(output)


if __name__ == '__main__':
    run(main)
