import abc
import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from clize import parameters

import slaid.commons.ecvl as ecvl
from slaid.classifiers import BasicClassifier, FilteredPatchClassifier
from slaid.commons.base import do_filter
from slaid.commons.factory import SlideFactory
from slaid.models.factory import Factory as ModelFactory
from slaid.writers import REGISTRY as STORAGE


def serial_basic(input_path: str, ):
    ...


@dataclass
class Runner(abc.ABC):
    input_path: str
    output_dir: str
    model_name: str
    level: int
    label: str
    writer: str
    threshold: float = None
    gpu: List[int] = None
    filter_: str = None
    overwrite_output_if_exists: bool = False
    no_round: bool = False
    filter_slide: str = None
    slide_reader: str = None
    batch_size: int = None

    def __post_init__(self):
        self.model = ModelFactory(self.model_name,
                                  gpu=self.gpu,
                                  batch=self.batch_size).get_model()
        _prepare_output_dir(self.output_dir)
        self.gpu = _convert_gpu_params(self.gpu)
        self.filter_ = _process_filter(self.filter_, self.filter_slide,
                                       self.slide_reader)
        self._classifier = None
        self._tile_size = None

    @abc.abstractproperty
    def classifier(self):
        ...

    def run(self):
        classifiled_slides = []
        for slide in _get_slides(self.input_path, self.slide_reader,
                                 self._tile_size):
            mask = self.classifier.classify(
                slide,
                self.filter_,
                self.threshold,
                self.level,
                round_to_0_100=not self.no_round,
            )
            slide.masks[self.label] = mask

            output_path = os.path.join(
                self.output_dir,
                f'{os.path.basename(slide.filename)}.{self.writer}')
            STORAGE[self.writer].dump(
                slide,
                output_path,
                overwrite=self.overwrite_output_if_exists,
                mask_name=self.label)
            logging.info('output %s', output_path)

            classifiled_slides.append(slide)
        return self.classifier, classifiled_slides


@dataclass
class SerialRunner(Runner):
    chunk_size: int = None

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = BasicClassifier(
                self.model,
                self.label,
                chunk=(self.chunk_size,
                       self.chunk_size) if self.chunk_size else None)
        return self._classifier


@dataclass
class SerialPatchRunner(Runner):

    def __post_init__(self):
        super().__post_init__()
        if self.model.patch_size is None:
            raise RuntimeError(
                f'model {self.model_name} does not work with patch')
        self._tile_size = self.model.patch_size[0]

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = FilteredPatchClassifier(self.model, self.label)
        return self._classifier


def serial(input_path: str,
           *,
           model: (str, 'm'),
           level: (int, 'l'),
           output_dir: (str, 'o'),
           label: (str, 'L'),
           threshold: (float, 't') = None,
           gpu: (int, parameters.multi()) = None,
           writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
               STORAGE.keys())[0],
           filter_: 'F' = None,
           overwrite_output_if_exists: 'overwrite' = False,
           no_round: bool = False,
           filter_slide: str = None,
           slide_reader: ('r', parameters.one_of('ecvl',
                                                 'openslide')) = 'ecvl',
           batch_size: ('b', int) = None,
           chunk_size: int = None):
    return SerialRunner(input_path,
                        model_name=model,
                        level=level,
                        output_dir=output_dir,
                        label=label,
                        threshold=threshold,
                        gpu=gpu,
                        writer=writer,
                        filter_=filter_,
                        overwrite_output_if_exists=overwrite_output_if_exists,
                        no_round=no_round,
                        filter_slide=filter_slide,
                        slide_reader=slide_reader,
                        batch_size=batch_size,
                        chunk_size=chunk_size).run()


def serial_patch(input_path: str,
                 *,
                 model: (str, 'm'),
                 level: (int, 'l'),
                 output_dir: (str, 'o'),
                 label: (str, 'L'),
                 threshold: (float, 't') = None,
                 gpu: (int, parameters.multi()) = None,
                 writer: ('w',
                          parameters.one_of(*list(STORAGE.keys()))) = list(
                              STORAGE.keys())[0],
                 filter_: 'F' = None,
                 overwrite_output_if_exists: 'overwrite' = False,
                 no_round: bool = False,
                 filter_slide: str = None,
                 slide_reader: ('r', parameters.one_of('ecvl',
                                                       'openslide')) = 'ecvl',
                 batch_size: ('b', int) = None):
    return SerialPatchRunner(
        input_path,
        model_name=model,
        level=level,
        output_dir=output_dir,
        label=label,
        threshold=threshold,
        gpu=gpu,
        writer=writer,
        filter_=filter_,
        overwrite_output_if_exists=overwrite_output_if_exists,
        no_round=no_round,
        filter_slide=filter_slide,
        slide_reader=slide_reader,
        batch_size=batch_size).run()


def _process_filter(filter_, filter_slide, slide_reader):
    if filter_:
        filter_slide = SlideFactory(
            filter_slide,
            slide_reader,
            'base',
        ).get_slide()
        filter_ = do_filter(filter_slide, filter_)
    return filter_


def _convert_gpu_params(gpu: List[int]) -> List[int]:
    if gpu:
        res = np.zeros(max(gpu) + 1, dtype='uint8')
        np.put(res, gpu, 1)
        return list(res)
    return gpu


def _prepare_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def _get_slides(input_path, slide_reader, tile_size):

    inputs = [
        os.path.abspath(os.path.join(input_path, f))
        for f in os.listdir(input_path)
    ] if os.path.isdir(input_path) and os.path.splitext(
        input_path)[-1][1:] not in STORAGE.keys() else [input_path]
    logging.info('processing inputs %s', inputs)
    for f in inputs:
        yield SlideFactory(f, slide_reader, 'base', tile_size).get_slide()

    def _get_slide(path, slide_reader, tile_size):
        return SlideFactory(path, slide_reader, 'base', tile_size).get_slide()
