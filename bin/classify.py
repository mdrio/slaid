#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import pickle

import pkg_resources
import tiledb
from clize import parameters, run

import slaid.commons.ecvl as ecvl
import slaid.writers.tiledb as tiledb_io
import slaid.writers.zarr as zarr_io
from slaid.classifiers import BasicClassifier, do_filter
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons.dask import init_client

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                    '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

STORAGE = {'zarr': zarr_io, 'tiledb': tiledb_io}


def set_model(func, model):
    def wrapper(*args, **kwargs):
        return func(*args, model=model, **kwargs)

    return wrapper


def set_feature(func, feature):
    def wrapper(*args, **kwargs):
        return func(*args, feature=feature, **kwargs)

    return wrapper


def get_parallel_classifier(model, feature):
    from slaid.classifiers.dask import Classifier, init_client
    init_client()
    return Classifier(model, feature)


def load_config_file(config_file: str, backend: str):
    if config_file is None:
        return
    if backend == 'tiledb':
        config = tiledb.Config.load(config_file)
        tiledb.Ctx(config)
        tiledb.VFS(config)


class SerialRunner:
    @classmethod
    def run(cls,
            input_path,
            output_dir,
            *,
            model: 'm',
            n_batch: ('b', int) = 1,
            extraction_level: ('l', int) = 2,
            feature: 'f',
            threshold: ('t', float) = None,
            gpu=False,
            writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
                STORAGE.keys())[0],
            filter_: 'F' = None,
            overwrite_output_if_exists: 'overwrite' = False,
            round_to_zero: ('z', float) = 0.01,
            config_file: str = None):
        classifier = cls.get_classifier(model, feature, gpu)
        cls.prepare_output_dir(output_dir)

        cls.classify_slides(input_path, output_dir, classifier, n_batch,
                            extraction_level, threshold, writer, filter_,
                            overwrite_output_if_exists, round_to_zero)

    @classmethod
    def get_classifier(cls, model, feature, gpu):
        model = cls.get_model(model, gpu)
        return BasicClassifier(model, feature)

    @staticmethod
    def get_model(filename, gpu):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        model.gpu = gpu
        return model

    @staticmethod
    def prepare_output_dir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def get_slides(input_path):
        inputs = [
            os.path.abspath(os.path.join(input_path, f))
            for f in os.listdir(input_path)
        ] if os.path.isdir(input_path) and os.path.splitext(
            input_path)[-1][1:] not in STORAGE.keys() else [input_path]
        logging.info('processing inputs %s', inputs)
        for f in inputs:
            slide_ext_with_dot = os.path.splitext(f)[-1]
            slide_ext = slide_ext_with_dot[1:]
            try:
                slide = STORAGE.get(slide_ext, ecvl).load(f)

            except Exception as ex:
                logging.error(f'an error occurs with file {f}: {ex}')
            else:
                yield slide

        #  return [
        #      os.path.join(input_path, f) for f in os.listdir(input_path)
        #  ] if os.path.isdir(input_path) and os.path.splitext(
        #      input_path)[-1][1:] not in STORAGE.keys() else [input_path]

    @classmethod
    def classify_slides(cls, input_path, output_dir, classifier, n_batch,
                        extraction_level, threshold, writer, filter_,
                        overwrite_output_if_exists, round_to_zero):

        for slide in cls.get_slides(input_path):
            cls.classify_slide(slide, output_dir, classifier, n_batch,
                               extraction_level, threshold, writer, filter_,
                               overwrite_output_if_exists, round_to_zero)

    @classmethod
    def classify_slide(cls,
                       slide,
                       output_dir,
                       classifier,
                       n_batch,
                       extraction_level,
                       threshold=None,
                       writer=list(STORAGE.keys())[0],
                       filter_=None,
                       overwrite_output_if_exists=True,
                       round_to_zero=0.01):

        filter_ = do_filter(slide, filter_) if filter_ else None
        output_path = os.path.join(
            output_dir, f'{os.path.basename(slide.filename)}.{writer}')
        if classifier.feature in slide.masks or STORAGE[writer].mask_exists(
                output_path, classifier.feature):
            if not overwrite_output_if_exists:
                logging.info(
                    'skipping slide %s, feature %s already exists. '
                    'See flag overwrite', slide.filename, classifier.feature)
                return

        mask = classifier.classify(slide,
                                   n_batch=n_batch,
                                   filter_=filter_,
                                   threshold=threshold,
                                   level=extraction_level,
                                   round_to_zero=round_to_zero)
        feature = classifier.feature
        slide.masks[feature] = mask
        STORAGE[writer].dump(slide,
                             output_path,
                             overwrite=overwrite_output_if_exists,
                             mask=feature)
        logging.info('output %s', output_path)

    @staticmethod
    def get_output_filename(slide_filename, output_dir, ext):
        slide_basename = os.path.basename(slide_filename)
        output_filename = f'{slide_basename}.{ext}'
        output_filename = os.path.join(output_dir, output_filename)
        return output_filename


class ParallelRunner(SerialRunner):
    @classmethod
    def run(cls,
            input_path,
            output_dir,
            *,
            model: 'm',
            n_batch: ('b', int) = 1,
            processes: 'p' = False,
            extraction_level: ('l', int) = 2,
            feature: 'f',
            threshold: ('t', float) = None,
            gpu=False,
            writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
                STORAGE.keys())[0],
            filter_: 'F' = None,
            overwrite_output_if_exists: 'overwrite' = False,
            round_to_zero: ('z', float) = 0.01):
        classifier = cls.get_classifier(model, feature, gpu, processes)
        cls.prepare_output_dir(output_dir)

        cls.classify_slides(input_path, output_dir, classifier, n_batch,
                            extraction_level, threshold, writer, filter_,
                            overwrite_output_if_exists, round_to_zero)

    @classmethod
    def get_classifier(cls, model, feature, gpu, processes):
        model = cls.get_model(model, gpu)
        init_client(processes=processes)
        return DaskClassifier(model, feature)


if __name__ == '__main__':

    runners = {'serial': SerialRunner.run, 'parallel': ParallelRunner.run}
    model = os.environ.get("SLAID_MODEL")
    if model:
        model = pkg_resources.resource_filename('slaid',
                                                f'resources/models/{model}')
        for k, v in runners.items():
            runners[k] = set_model(v, model)

    feature = os.environ.get("SLAID_FEATURE")
    if feature:
        for k, v in runners.items():
            runners[k] = set_feature(v, feature)

    run(runners)
