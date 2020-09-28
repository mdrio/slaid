#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pickle

import pkg_resources
from clize import run

from slaid.classifiers import BasicClassifier
from slaid.commons import PATCH_SIZE
from slaid.commons.ecvl import create_slide
from slaid.renderers import to_json

logging.basicConfig(level=logging.INFO)


def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


WRITERS = {'json': to_json, 'pkl': pickle_dump}


def set_model(func, model):
    def wrapper(*args, **kwargs):
        return func(*args, model=model, **kwargs)

    return wrapper


def set_feature(func, feature):
    def wrapper(*args, **kwargs):
        return func(*args, feature=feature, **kwargs)

    return wrapper


def main(
    input_path,
    output_dir,
    *,
    model: 'm',
    extraction_level: ('l', int) = 2,
    feature: 'f',
    threshold: 't' = 0.8,
    patch_size=f'{PATCH_SIZE[0]}x{PATCH_SIZE[1]}',
    gpu=False,
    only_mask=False,
    writer: 'w' = 'pkl',
    filter_: 'F' = None,
    overwrite_output_if_exists: 'overwrite' = False,
    skip_output_if_exist=False,
):
    patch_size = tuple([int(e) for e in patch_size.split('x')])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    slides = [os.path.join(input_path, f) for f in os.listdir(input_path)
              ] if os.path.isdir(input_path) else [input_path]

    for slide in slides:
        classify_slide(slide, output_dir, model, feature, extraction_level,
                       threshold, patch_size, gpu, only_mask, writer, filter_,
                       overwrite_output_if_exists, skip_output_if_exist)


def classify_slide(slide_filename,
                   output_dir,
                   model,
                   feature,
                   extraction_level,
                   threshold=0.8,
                   patch_size=PATCH_SIZE,
                   gpu=False,
                   only_mask=False,
                   writer='pkl',
                   filter_=None,
                   overwrite_output_if_exists=True,
                   skip_output_if_exist=False):

    output_filename = get_output_filename(slide_filename, output_dir, writer,
                                          feature)
    if os.path.exists(output_filename):
        if skip_output_if_exist:
            logging.debug(f"""
                Skipping classification of slide {slide_filename},
                already exists.
                """)
            return

        elif not overwrite_output_if_exists:
            raise RuntimeError(f"""
                output for slide {slide_filename} already exists.
                Set parameter skip_output_if_exist to skip
                this slide classification or
                overwrite_output_if_exists to overwrite.
                """)

    slide_ext_with_dot = os.path.splitext(slide_filename)[-1]
    slide_ext = slide_ext_with_dot[1:]
    if slide_ext == 'pkl' or slide_ext == 'pickle':
        with open(slide_filename, 'rb') as f:
            slide = pickle.load(f)
    else:
        slide = create_slide(slide_filename)

    if os.path.splitext(model)[-1] in ('.pkl', '.pickle'):
        from slaid.models import PickledModel
        model = PickledModel(model)
    else:
        from slaid.models.eddl import Model
        model = Model(model, gpu)

    tissue_classifier = BasicClassifier(model, feature)

    tissue_classifier.classify(slide,
                               patch_filter=filter_,
                               threshold=threshold,
                               level=extraction_level)
    if only_mask:
        data_to_dump = {
            'filename': slide_filename,
            'dimensions': slide.dimensions,
            'extraction_level': slide.masks[feature].extraction_level,
            'mask': slide.masks[feature].array
        }

    else:
        data_to_dump = slide

    WRITERS[writer](data_to_dump, output_filename)
    logging.info(output_filename)


def get_output_filename(slide_filename, output_dir, ext, feature):
    slide_basename = os.path.basename(slide_filename)
    slide_basename_no_ext = os.path.splitext(slide_basename)[0]
    output_filename = f'{slide_basename_no_ext}.{feature}.{ext}'
    output_filename = os.path.join(output_dir, output_filename)
    return output_filename


if __name__ == '__main__':

    model = os.environ.get("SLAID_MODEL")
    if model is not None:
        model = pkg_resources.resource_filename('slaid', f'models/{model}')
        main = set_model(main, model)
    feature = os.environ.get("SLAID_FEATURE")
    if feature is not None:
        main = set_feature(main, feature)
    run(main)
