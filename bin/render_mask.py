#!/usr/bin/env python
# -*- coding: utf-8 -*-
from clize import run
from slaid.writers import REGISTRY


def main(archive,
         mask_name,
         output,
         *,
         downsample: ('d', int) = 1,
         threshold: (float, 't') = None):

    slide = REGISTRY[archive.splitext()[1:]].load(archive)
    mask = slide.masks[mask_name]
    mask.to_image(downsample, threshold).save(output)


if __name__ == '__main__':
    run(main)
