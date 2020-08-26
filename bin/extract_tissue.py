#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle

import pkg_resources

from slaid.classifiers import (BasicTissueClassifier, BasicTissueMaskPredictor,
                               get_tissue_mask)
from slaid.commons import PATCH_SIZE
from slaid.commons.ecvl import Slide
from slaid.renderers import PickleRenderer


def main(slide_filename,
         model_filename,
         output_filename,
         extraction_level,
         pixel_threshold=0.8,
         minimum_tissue_ratio=0.1,
         include_mask=True,
         patch_size=PATCH_SIZE,
         gpu=False,
         only_mask=False):
    slide = Slide(slide_filename,
                  patch_size=patch_size,
                  extraction_level=extraction_level)

    if os.path.splitext(model_filename)[-1] in ('.pkl', '.pickle'):
        from slaid.classifiers import Model
        model = Model(model_filename)
    else:
        from slaid.classifiers.eddl import Model
        model = Model(model_filename, gpu)

    tissue_classifier = BasicTissueClassifier(BasicTissueMaskPredictor(model))

    tissue_classifier.classify(slide,
                               pixel_threshold=pixel_threshold,
                               minimum_tissue_ratio=minimum_tissue_ratio,
                               include_mask_feature=include_mask)

    if only_mask:
        with open(output_filename, 'wb') as f:
            pickle.dump(
                {
                    'filename': slide_filename,
                    'dimensions': slide.dimensions,
                    'extraction_level': extraction_level,
                    'mask': get_tissue_mask(slide)
                }, f)

    else:
        pickle_renderer = PickleRenderer()
        pickle_renderer.render(output_filename, slide)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('slide')
    parser.add_argument('output')
    # workaround since is not possible to pass env variable
    # to Docker CMD
    model = os.environ.get("SLAID_MODEL")
    if model is None:
        parser.add_argument(
            '-m',
            dest='model',
            help='path to model',
        )
    else:
        model = pkg_resources.resource_filename('slaid', f'models/{model}')

    parser.add_argument('--patch_size', dest='patch_size', default=PATCH_SIZE)
    parser.add_argument('-l',
                        dest='extraction_level',
                        help='extraction_level',
                        default=2,
                        type=int)
    parser.add_argument('-t',
                        dest='pixel_threshold',
                        default=0.8,
                        help="pixel pixel threshold",
                        type=float)
    parser.add_argument('-T',
                        dest='minimum_tissue_ratio',
                        default=0.1,
                        help="minimum tissue ratio",
                        type=float)
    parser.add_argument('--no-mask',
                        dest='no_mask',
                        default=False,
                        help="not include tissue mask",
                        action='store_true')
    parser.add_argument('--only-mask',
                        dest='only_mask',
                        default=False,
                        help="only tissue mask is returned",
                        action='store_true')
    parser.add_argument('--gpu',
                        dest='gpu',
                        default=False,
                        help="use gpu",
                        action='store_true')

    args = parser.parse_args()
    if model is None:
        model = args.model
    main(args.slide, model, args.output, args.extraction_level,
         args.pixel_threshold, args.minimum_tissue_ratio, not args.no_mask,
         args.patch_size, args.gpu, args.only_mask)
