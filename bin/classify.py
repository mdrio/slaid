#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import pkg_resources

from slaid.classifiers import BasicClassifier
from slaid.commons import PATCH_SIZE
from slaid.commons.ecvl import create_slide
from slaid.renderers import to_json


def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


WRITERS = {'json': to_json, 'pkl': pickle_dump}


def main(input_path,
         output_dir,
         model_filename,
         feature,
         extraction_level,
         pixel_threshold=0.8,
         patch_threshold=0.5,
         include_mask=True,
         patch_size=PATCH_SIZE,
         gpu=False,
         only_mask=False,
         writer='json',
         filter_=None):

    slides = [f for f in os.listdir(input_path)
              ] if os.path.isdir(input_path) else [input_path]

    for slide in slides:
        classify_slide(slide, output_dir, model_filename, feature,
                       extraction_level, pixel_threshold, patch_threshold,
                       include_mask, patch_size, gpu, only_mask, writer,
                       filter_)


def classify_slide(
    slide_filename,
    output_dir,
    model_filename,
    feature,
    extraction_level,
    pixel_threshold=0.8,
    patch_threshold=0.5,
    include_mask=True,
    patch_size=PATCH_SIZE,
    gpu=False,
    only_mask=False,
    writer='pkl',
    filter_=None,
):
    slide_ext_with_dot = os.path.splitext(slide_filename)[-1]
    slide_ext = slide_ext_with_dot[1:]

    if slide_ext == 'pkl' or slide_ext == 'pickle':
        with open(slide_filename, 'rb') as f:
            slide = pickle.load(f)
    else:
        slide = create_slide(slide_filename,
                             extraction_level=extraction_level,
                             patch_size=patch_size)

    if os.path.splitext(model_filename)[-1] in ('.pkl', '.pickle'):
        from slaid.classifiers import Model
        model = Model(model_filename)
    else:
        from slaid.classifiers.eddl import Model
        model = Model(model_filename, gpu)

    tissue_classifier = BasicClassifier(model, feature)

    tissue_classifier.classify(slide,
                               patch_filter=filter_,
                               mask_threshold=pixel_threshold,
                               patch_threshold=patch_threshold,
                               include_mask=include_mask)

    slide_basename = os.path.basename(slide_filename)
    slide_basename_no_ext = os.path.splitext(slide_basename)[0]
    output_filename = f'{slide_basename_no_ext}.{feature}.{writer}'
    ext_with_dot = os.path.splitext(output_filename)[-1]
    ext = ext_with_dot[1:]

    if only_mask:
        data_to_dump = {
            'filename': slide_filename,
            'dimensions': slide.dimensions,
            'extraction_level': extraction_level,
            'mask': slide.masks['tissue']
        }

    else:
        data_to_dump = slide

    output_filename = os.path.join(output_dir, output_filename)
    WRITERS[ext](data_to_dump, output_filename)
    print(output_filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('slide')
    parser.add_argument('output_dir')
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

    feature = os.environ.get("SLAID_FEATURE")
    if feature is None:
        parser.add_argument('-f', dest='feature', help="feature label")

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
    parser.add_argument('-w',
                        dest='writer',
                        default='pkl',
                        help="writer for serializing the resulting output",
                        choices=WRITERS.keys())

    parser.add_argument(
        '-F',
        dest='filter',
        default=None,
        help="filter by patch feature",
    )
    args = parser.parse_args()
    model = model or args.model
    feature = feature or args.feature

    main(args.slide, args.output_dir, model, feature, args.extraction_level,
         args.pixel_threshold, args.minimum_tissue_ratio, not args.no_mask,
         args.patch_size, args.gpu, args.only_mask, args.writer, args.filter)
