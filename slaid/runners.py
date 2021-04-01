import logging
import os
import pickle
from importlib import import_module
from typing import List

import numpy as np
from clize import parameters

import slaid.commons.ecvl as ecvl
import slaid.writers.tiledb as tiledb_io
import slaid.writers.zarr as zarr_io
from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons.base import Slide, SlideStore, do_filter
from slaid.commons.dask import DaskSlide, init_client
from slaid.models.dask import ActorModel
from slaid.models.eddl import load_model

STORAGE = {'zarr': zarr_io, 'tiledb': tiledb_io}


class SerialRunner:
    CLASSIFIER = BasicClassifier

    @staticmethod
    def get_slide(path, slide_reader):
        slide_ext_with_dot = os.path.splitext(path)[-1]
        slide_ext = slide_ext_with_dot[1:]
        try:
            return Slide(
                SlideStore(STORAGE.get(slide_ext, slide_reader).load(path)))
        except Exception as ex:
            logging.error('an error occurs with file %s: %s', path, ex)

    @staticmethod
    def convert_gpu_params(gpu: List[int]) -> List[int]:
        if gpu:
            res = np.zeros(max(gpu) + 1, dtype='uint8')
            np.put(res, gpu, 1)
            return list(res)
        return gpu

    @classmethod
    def run(cls,
            input_path,
            *,
            output_dir: 'o',
            model: 'm',
            max_MB_prediction: ('b', float) = None,
            extraction_level: ('l', int) = 2,
            feature: 'f',
            threshold: ('t', float) = None,
            gpu: (int, parameters.multi()) = None,
            writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
                STORAGE.keys())[0],
            filter_: 'F' = None,
            overwrite_output_if_exists: 'overwrite' = False,
            no_round: bool = False,
            filter_slide: str = None,
            slide_reader: ('r', parameters.one_of('ecvl',
                                                  'openslide')) = 'ecvl',
            dry_run: bool = False):
        if dry_run:
            args = dict(locals())
            args.pop('cls')
            print(args)
        else:
            gpu = cls.convert_gpu_params(gpu)
            classifier = cls.get_classifier(model, feature, gpu)
            cls.prepare_output_dir(output_dir)

            slides = cls.classify_slides(input_path, output_dir, classifier,
                                         max_MB_prediction, extraction_level,
                                         threshold, writer, filter_,
                                         overwrite_output_if_exists, no_round,
                                         filter_slide, slide_reader)
            return classifier, slides

    @classmethod
    def get_classifier(cls, model, feature, gpu):
        model = cls.get_model(model, gpu)
        return cls.CLASSIFIER(model, feature)

    @staticmethod
    def get_model(filename, gpu):
        ext = os.path.splitext(filename)[-1]
        if ext in ('.pkl', '.pickle'):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        elif ext == '.bin':
            model = load_model(filename)
        else:
            raise NotImplementedError(f'unsupported model type {ext}')
        model.gpu = gpu
        return model

    @staticmethod
    def prepare_output_dir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def get_slides(cls, input_path, slide_reader):

        inputs = [
            os.path.abspath(os.path.join(input_path, f))
            for f in os.listdir(input_path)
        ] if os.path.isdir(input_path) and os.path.splitext(
            input_path)[-1][1:] not in STORAGE.keys() else [input_path]
        logging.info('processing inputs %s', inputs)
        for f in inputs:
            yield cls.get_slide(f, slide_reader)

    @classmethod
    def classify_slides(cls, input_path, output_dir, classifier,
                        max_MB_prediction, extraction_level, threshold, writer,
                        filter_, overwrite_output_if_exists, no_round,
                        filter_slide, slide_reader):

        slides = []
        slide_reader = import_module(f'slaid.commons.{slide_reader}')
        for slide in cls.get_slides(input_path, slide_reader):
            cls.classify_slide(slide, output_dir, classifier,
                               max_MB_prediction, extraction_level,
                               slide_reader, threshold, writer, filter_,
                               overwrite_output_if_exists, no_round,
                               filter_slide)
            slides.append(slide)
        return slides

    @classmethod
    def classify_slide(cls,
                       slide,
                       output_dir,
                       classifier,
                       max_MB_prediction,
                       extraction_level,
                       slide_reader,
                       threshold=None,
                       writer=list(STORAGE.keys())[0],
                       filter_=None,
                       overwrite_output_if_exists=True,
                       no_round: bool = False,
                       filter_slide=None):

        if filter_:
            filter_slide = get_slide(filter_slide,
                                     slide_reader) if filter_slide else slide
            filter_ = do_filter(filter_slide, filter_)
        output_path = os.path.join(
            output_dir, f'{os.path.basename(slide.filename)}.{writer}')
        if classifier.feature in slide.masks or STORAGE[writer].mask_exists(
                output_path, classifier.feature):
            if not overwrite_output_if_exists:
                logging.info(
                    'skipping slide %s, feature %s already exists. '
                    'See flag overwrite', slide.filename, classifier.feature)
                return slide

        mask = classifier.classify(slide,
                                   max_MB_prediction=max_MB_prediction,
                                   filter_=filter_,
                                   threshold=threshold,
                                   level=extraction_level,
                                   round_to_0_100=not no_round)
        feature = classifier.feature
        slide.masks[feature] = mask
        STORAGE[writer].dump(slide,
                             output_path,
                             overwrite=overwrite_output_if_exists,
                             mask=feature)
        logging.info('output %s', output_path)
        return slide

    @staticmethod
    def get_output_filename(slide_filename, output_dir, ext):
        slide_basename = os.path.basename(slide_filename)
        output_filename = f'{slide_basename}.{ext}'
        output_filename = os.path.join(output_dir, output_filename)
        return output_filename


class ParallelRunner(SerialRunner):
    CLASSIFIER = DaskClassifier

    @classmethod
    def run(cls,
            input_path,
            *,
            processes: 'p' = False,
            scheduler: str = None,
            output_dir: 'o',
            model: 'm',
            max_MB_prediction: ('b', float) = None,
            extraction_level: ('l', int) = 2,
            feature: 'f',
            threshold: ('t', float) = None,
            gpu: (int, parameters.multi()) = None,
            writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
                STORAGE.keys())[0],
            filter_: 'F' = None,
            overwrite_output_if_exists: 'overwrite' = False,
            no_round: bool = False,
            filter_slide: str = None,
            slide_reader: ('r', parameters.one_of('ecvl',
                                                  'openslide')) = 'ecvl',
            dry_run: bool = False):
        kwargs = dict(locals())
        for key in ('cls', '__class__', 'processes', 'scheduler'):
            kwargs.pop(key)
        cls._init_client(scheduler, processes)
        return super().run(**kwargs)

    @staticmethod
    def _init_client(scheduler, processes):
        init_client(address=scheduler, processes=processes)

    @staticmethod
    def get_slide(path, slide_reader):
        slide_ext_with_dot = os.path.splitext(path)[-1]
        slide_ext = slide_ext_with_dot[1:]
        try:
            return DaskSlide(
                SlideStore(STORAGE.get(slide_ext, slide_reader).load(path)))
        except Exception as ex:
            logging.error('an error occurs with file %s: %s', path, ex)

    @staticmethod
    def get_model(filename, gpu):
        return ActorModel.create(SerialRunner.get_model,
                                 filename=filename,
                                 gpu=gpu)
