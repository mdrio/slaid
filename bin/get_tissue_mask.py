#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

import pkg_resources

import slaid
from slaid.classifiers import BasicTissueClassifier, get_tissue_mask
from slaid.commons import PATCH_SIZE, Slide, UniqueStore


def main(slide_filename,
         model_filename,
         output_filename,
         extraction_level,
         pixel_threshold=0.8,
         minimum_tissue_ratio=0.1,
         patch_size=PATCH_SIZE):
    slide = Slide(slide_filename,
                  patch_size=patch_size,
                  extraction_level=extraction_level)

    tissue_classifier = BasicTissueClassifier.create(model_filename)

    tissue_classifier.classify(slide,
                               pixel_threshold=pixel_threshold,
                               minimum_tissue_ratio=minimum_tissue_ratio,
                               include_mask_feature=True)
    with open(output_filename, 'wb') as f:
        pickle.dump(
            {
                'filename': slide_filename,
                'dimensions': slide.dimensions,
                'extraction_level': extraction_level,
                'mask': get_tissue_mask(slide)
            }, f)


if __name__ == '__main__':
    import argparse

    default_model = pkg_resources.resource_filename(
        slaid.__name__, 'models/extract_tissue_LSVM-1.0.pickle')
    parser = argparse.ArgumentParser()
    parser.add_argument('slide')
    parser.add_argument('output')
    parser.add_argument('-m',
                        dest='model',
                        help='path to model',
                        action=UniqueStore,
                        default=default_model)
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

    args = parser.parse_args()
    main(args.slide, args.model, args.output, args.extraction_level,
         args.pixel_threshold, args.minimum_tissue_ratio, args.patch_size)
