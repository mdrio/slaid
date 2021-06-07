import logging
import os
from typing import List, Tuple

import numpy as np
from clize import parameters

import slaid.commons.ecvl as ecvl
import slaid.writers.tiledb as tiledb_io
import slaid.writers.zarr as zarr_io
from slaid.classifiers import BasicClassifier
from slaid.classifiers.dask import Classifier as DaskClassifier
from slaid.commons.base import DEFAULT_TILESIZE, do_filter
from slaid.commons.dask import init_client
from slaid.commons.factory import SlideFactory
from slaid.models.factory import Factory as ModelFactory

STORAGE = {'zarr': zarr_io, 'tiledb': tiledb_io}


class SerialRunner:
    CLASSIFIER = BasicClassifier

    @staticmethod
    def get_slide(path, slide_reader, tilesize):
        try:
            return SlideFactory(path, slide_reader, 'base').get_slide()
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
            chunk: int = None,
            slide_reader: ('r', parameters.one_of('ecvl',
                                                  'openslide')) = 'ecvl',
            batch: ('b', int) = None,
            dry_run: bool = False):
        """
        :param batch: how many bytes will be predicted at once. Default: all chunk is predicted (see chunk)
        :param chunk: the size (square) of data processed at once.

        """
        if chunk:
            chunk = (chunk, chunk)
            tilesize = chunk
        else:
            tilesize = DEFAULT_TILESIZE

        if dry_run:
            args = dict(locals())
            args.pop('cls')
            print(args)
        else:
            gpu = cls.convert_gpu_params(gpu)
            classifier = cls.get_classifier(model, feature, gpu, batch, writer)
            cls.prepare_output_dir(output_dir)
            slides = cls.classify_slides(input_path, output_dir, classifier,
                                         extraction_level, threshold, writer,
                                         filter_, overwrite_output_if_exists,
                                         no_round, filter_slide, chunk,
                                         slide_reader, tilesize)
            return classifier, slides

    @classmethod
    def get_classifier(cls,
                       model,
                       feature,
                       gpu,
                       batch,
                       writer=list(STORAGE.keys())[0]):
        model = ModelFactory(model, gpu=gpu, batch=batch).get_model()
        return cls.CLASSIFIER(model, feature, STORAGE[writer].empty)

    @staticmethod
    def prepare_output_dir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def get_slides(cls, input_path, slide_reader, tilesize):

        inputs = [
            os.path.abspath(os.path.join(input_path, f))
            for f in os.listdir(input_path)
        ] if os.path.isdir(input_path) and os.path.splitext(
            input_path)[-1][1:] not in STORAGE.keys() else [input_path]
        logging.info('processing inputs %s', inputs)
        for f in inputs:
            yield cls.get_slide(f, slide_reader, tilesize)

    @classmethod
    def classify_slides(cls, input_path, output_dir, classifier,
                        extraction_level, threshold, writer, filter_,
                        overwrite_output_if_exists, no_round, filter_slide,
                        chunk, slide_reader, tilesize):

        slides = []
        for slide in cls.get_slides(input_path, slide_reader, tilesize):
            cls.classify_slide(slide, output_dir, classifier, extraction_level,
                               slide_reader, threshold, writer, filter_,
                               overwrite_output_if_exists, no_round,
                               filter_slide, chunk)
            slides.append(slide)
        return slides

    @classmethod
    def classify_slide(cls,
                       slide,
                       output_dir,
                       classifier,
                       extraction_level,
                       slide_reader,
                       threshold=None,
                       writer=list(STORAGE.keys())[0],
                       filter_=None,
                       overwrite_output_if_exists=True,
                       no_round: bool = False,
                       filter_slide=None,
                       chunk=None):

        if filter_:
            filter_slide = cls.get_slide(filter_slide, slide_reader, 256)
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

        chunk = chunk or (-1, -1)
        tmp_chunk = list(chunk)
        for i in range(len(chunk)):
            if chunk[i] < 0:
                tmp_chunk[i] = slide.level_dimensions[extraction_level][::-1][
                    i]

        chunk = tuple(tmp_chunk)
        mask = classifier.classify(slide,
                                   filter_=filter_,
                                   threshold=threshold,
                                   level=extraction_level,
                                   round_to_0_100=not no_round,
                                   chunk=chunk)
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
            chunk: int = None,
            batch: ('b', int) = None,
            dry_run: bool = False):
        kwargs = dict(locals())
        for key in ('cls', '__class__', 'processes', 'scheduler'):
            kwargs.pop(key)
        cls._init_client(scheduler, processes)
        return super().run(**kwargs)

    @classmethod
    def get_classifier(cls,
                       model,
                       feature,
                       gpu,
                       batch,
                       writer=list(STORAGE.keys())[0]):
        model = ModelFactory(model, gpu=gpu, batch=batch).get_model()
        return cls.CLASSIFIER(model, feature)

    @staticmethod
    def _init_client(scheduler, processes):
        init_client(address=scheduler, processes=processes)

    @staticmethod
    def get_slide(path, slide_reader, tilesize):
        try:
            return SlideFactory(path, slide_reader, 'dask',
                                tilesize).get_slide()
        except Exception as ex:
            logging.error('an error occurs with file %s: %s', path, ex)
