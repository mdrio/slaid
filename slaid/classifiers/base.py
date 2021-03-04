import abc
import logging
import re
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Tuple

import numpy as np
from progress.bar import Bar

from slaid.commons import Mask, Slide
from slaid.commons.base import Image
from slaid.models import Model

logger = logging.getLogger('classify')


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True) -> Mask:
        pass


@dataclass
class Filter:
    mask: Mask

    def filter(self, operator: str, value: float) -> np.ndarray:
        mask = np.array(self.mask.array)
        if self.mask.round_to_0_100:
            mask = mask / 100
        index_patches = getattr(mask, operator)(value)
        return np.argwhere(index_patches)

    def __gt__(self, value):
        return self.filter('__gt__', value)

    def __ge__(self, value):
        return self.filter('__ge__', value)

    def __lt__(self, value):
        return self.filter('__lt__', value)

    def __le__(self, value):
        return self.filter('__le__', value)

    def __eq__(self, value):
        return self.filter('__eq__', value)

    def __ne__(self, value):
        return self.filter('__ne__', value)


def do_filter(slide: Slide, condition: str) -> "Filter":
    operator_mapping = {
        '>': '__gt__',
        '>=': '__ge__',
        '<': '__lt__',
        '<=': '__le__',
        '==': '__eq__',
        '!=': '__ne__',
    }
    parsed = re.match(
        r"(?P<mask>\w+)\s*(?P<operator>[<>=!]+)\s*(?P<value>\d+\.*\d*)",
        condition).groupdict()
    mask = slide.masks[parsed['mask']]
    operator = operator_mapping[parsed['operator']]
    value = float(parsed['value'])
    return Filter(mask).filter(operator, value)


class BasicClassifier(Classifier):
    MASK_CLASS = Mask

    def __init__(self, model: "Model", feature: str):
        self.model = model
        self.feature = feature
        self._patch_size = self.model.patch_size
        self._color_type = model.color_type
        self._coords = model.coords
        self._channel = model.channel

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True,
                 n_patch=25) -> Mask:

        logger.info('classify: %s, %s, %s, %s, %s, %s', slide.filename,
                    filter_, threshold, level, n_batch, round_to_0_100)
        batches = BatchIterator(slide, level, n_batch, self._color_type,
                                self._coords, self._channel)
        array = self._classify_patches(
            slide, self._patch_size, level, filter_, threshold, n_patch,
            round_to_0_100) if self._patch_size else self._classify_batches(
                batches, threshold, round_to_0_100)

        return self._get_mask(array, level, slide.level_downsamples[level],
                              dt.now(), round_to_0_100)

    def _get_mask(self, array, level, downsample, datetime, round_to_0_100):
        return self.MASK_CLASS(array,
                               level,
                               downsample,
                               datetime,
                               round_to_0_100,
                               model=str(self.model))

    def _classify_patches(self,
                          slide: Slide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25,
                          round_to_0_100: bool = True) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)

        patch_indexes = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        patches_to_predict = [
            Patch(slide, p[0], p[1], level, patch_size, self.model.color_type,
                  self.model.coords, self.model.channel) for p in patch_indexes
        ]

        # adding fake patches, workaround for
        # https://github.com/deephealthproject/eddl/issues/236
        #  for _ in range(patch_to_add):
        #      patches_to_predict.append(patches_to_predict[0])

        predictions = []
        with Bar('Predictions', max=len(patches_to_predict) // n_patch
                 or 1) as predict_bar:
            for i in range(0, len(patches_to_predict), n_patch):
                patches = patches_to_predict[i:i + n_patch]
                predictions.append(
                    self._classify_array(np.stack([p.array for p in patches]),
                                         threshold, round_to_0_100))
                predict_bar.next()
        if predictions:
            predictions = np.concatenate(predictions)
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch.row, patch.column] = p
        return res

    def _classify_batches(self, batches: "BatchIterator", threshold: float,
                          round_to_0_100: bool) -> Mask:
        predictions = []
        for batch in batches:
            prediction = self._classify_batch(batch, threshold, round_to_0_100)
            if prediction.size:
                predictions.append(prediction)
        return self._concatenate(predictions, axis=0)

    def _classify_batch(self, batch, threshold, round_to_0_100):
        # FIXME
        filter_ = None
        array = batch.array
        orig_shape = batch.shape
        if filter_ is not None:
            indexes_pixel_to_process = filter_.filter(batch)
            array = array[indexes_pixel_to_process]
        else:
            array = batch.flatten()

        prediction = self._classify_array(array, threshold, round_to_0_100)
        if filter_ is not None:
            res = np.zeros(orig_shape[:2], dtype=prediction.dtype)
            res[indexes_pixel_to_process] = prediction
        else:
            res = prediction.reshape(orig_shape[:2])
        return res

    def _classify_array(self, array, threshold, round_to_0_100) -> np.ndarray:
        prediction = self.model.predict(array)
        if round_to_0_100:
            prediction = prediction * 100
            return prediction.astype('uint8')
        if threshold:
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            return prediction.astype('uint8')

        return prediction.astype('float32')

    @staticmethod
    def _get_batch(slide, start, size, level):
        image = slide.read_region(start[::-1], level, size[::-1])
        array = image.to_array(True)
        return Batch(start, size, array, slide.level_downsamples[level])

    @staticmethod
    def _get_zeros(size, dtype):
        return np.zeros(size, dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return array.reshape(shape)

    def _flat_array(self, array: np.ndarray, shape: Tuple[int,
                                                          int]) -> np.ndarray:
        n_px = shape[0] * shape[1]
        array = array[:, :, :3].reshape(n_px, 3)
        return array


@dataclass
class ImageArea:
    slide: Slide
    row: int
    column: int
    level: int
    size: Tuple[int, int]
    color_type: Image.COLORTYPE = Image.COLORTYPE.BGR
    coords: Image.COORD = Image.COORD.YX
    channel: Image.CHANNEL = Image.CHANNEL.FIRST

    def __post_init__(self):
        self._array = None

    @property
    def top_level_location(self):
        raise NotImplementedError

    @property
    def array(self):
        if self._array is None:
            image = self.slide.read_region(self.top_level_location[::-1],
                                           self.level, self.size[::-1])
            self._array = image.to_array(self.color_type, self.coords,
                                         self.channel)
        return self._array

    @property
    def shape(self):
        return self.array.shape[:2] if self.channel == Image.CHANNEL.LAST \
            else self.array.shape[1:]

    def flatten(self):
        n_px = self.shape[0] * self.shape[1]
        if self.channel == Image.CHANNEL.FIRST:
            array = self.array.transpose(1, 2, 0)
        else:
            array = self.array
        array = array[:, :, :3].reshape(n_px, 3)
        return array


@dataclass
class Batch(ImageArea):
    def __post_init__(self):
        super().__post_init__()
        self.downsample = self.slide.level_downsamples[self.level]

    @property
    def top_level_location(self):
        return (round(self.row * self.downsample),
                round(self.column * self.downsample))

    def get_patches(self,
                    patch_size: Tuple[int, int],
                    filter_: Filter = None) -> "Patch":
        dimensions = self.slide.level_dimensions[self.level][::-1]
        if filter_ is None:
            for row in range(0, self.size[0], patch_size[0]):
                if self.row + row + patch_size[0] > dimensions[0]:
                    continue
                for col in range(0, self.size[1], patch_size[1]):
                    if self.column + col + patch_size[1] > dimensions[1]:
                        continue
                    patch_row = int(row // patch_size[0])
                    patch_column = int(col // patch_size[1])
                    yield Patch(self.slide, patch_row, patch_column,
                                patch_size, self.color_type, self.coords,
                                self.channel)

        else:
            index_patches = filter_.filter(self, patch_size)
            for p in np.argwhere(index_patches):
                patch = Patch(self.slide, p[0], p[1], patch_size,
                              self.color_type, self.coords, self.channel)
                logger.debug('yielding patch %s', patch)
                yield patch


@dataclass
class BatchIterator:
    slide: Slide
    level: int
    n_batch: int
    color_type: Image.COLORTYPE = Image.COLORTYPE.BGR
    coords: Image.COORD = Image.COORD.YX
    channel: Image.CHANNEL = Image.CHANNEL.FIRST

    def __post_init__(self):
        self._level_dimensions = self.slide.level_dimensions[self.level][::-1]
        self._downsample = self.slide.level_downsamples[self.level]

        batch_size_0 = self._level_dimensions[0] // self.n_batch
        self._batch_size = (batch_size_0, self._level_dimensions[1])

    def __iter__(self):
        for i in range(0, self._level_dimensions[0], self._batch_size[0]):
            size_0 = min(self._batch_size[0], self._level_dimensions[0] - i)
            for j in range(0, self._level_dimensions[1], self._batch_size[1]):
                size = (size_0,
                        min(self._batch_size[1],
                            self._level_dimensions[1] - j))
                yield Batch(self.slide, i, j, self.level, size,
                            self.color_type, self.coords, self.channel)

    def __len__(self):
        return self._level_dimensions[0] // self._batch_size[0]


@dataclass
class Patch(ImageArea):
    slide: Slide
    row: int
    column: int
    level: int
    size: Tuple[int, int]
    PIL_format: bool = False

    def __post_init__(self):
        self._array = None

    @property
    def top_level_location(self):
        return (self.row * self.size[0], self.column * self.size[1])
