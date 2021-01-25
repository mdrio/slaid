import abc
import logging
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from progress.bar import Bar

from slaid.commons import Mask, Slide
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
                 round_to_zero: float = 0.01) -> Mask:
        pass


@dataclass
class Filter:
    mask: Mask

    def filter(self, operator: str, value: float) -> np.ndarray:
        mask = np.array(self.mask.array)
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
    def __init__(self, model: "Model", feature: str):
        self.model = model
        self.feature = feature

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_zero: float = 0.01) -> Mask:

        patch_size = self.model.patch_size
        batches = BatchIterator(slide, level, n_batch)
        array = self._classify_patches(
            slide, patch_size, level, filter_,
            threshold) if patch_size else self._classify_batches(
                batches, threshold)

        #  mask = self._round_to_zero(mask, round_to_zero)
        return self._get_mask(array, level, slide.level_downsamples[level])

    @staticmethod
    def _get_mask(array, level, downsample):
        return Mask(array, level, downsample)

    def _classify_patches(self,
                          slide: Slide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold else 'float32'
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)

        patch_indexes = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        patches_to_predict = [
            Patch(slide, p[0], p[1], level, patch_size) for p in patch_indexes
        ]

        mod = len(patches_to_predict) % n_patch
        patch_to_add = n_patch - mod if mod else 0

        # adding fake patches, workaround for
        # https://github.com/deephealthproject/eddl/issues/236
        for _ in range(patch_to_add):
            patches_to_predict.append(patches_to_predict[0])

        predictions = []
        with Bar('Predictions', max=len(patches_to_predict) // n_patch
                 or 1) as predict_bar:
            for i in range(0, len(patches_to_predict), n_patch):
                patches = patches_to_predict[i:i + n_patch]
                predictions.append(
                    self._classify_array(np.stack([p.array for p in patches]),
                                         threshold))
                predict_bar.next()
        if predictions:
            predictions = np.concatenate(predictions)
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch.row, patch.column] = p
        return res

    def _round_to_zero(self, mask, threshold):
        if threshold > 0:
            mask[mask <= threshold] = 0
        return mask

    def _classify_batches(self, batches: "BatchIterator",
                          threshold: float) -> Mask:
        predictions = []
        for batch in batches:
            prediction = self._classify_batch(batch, threshold)
            if prediction.size:
                predictions.append(prediction)
        return self._concatenate(predictions, axis=0)

    def _classify_batch(self, batch, threshold):
        # FIXME
        filter_ = None
        array = batch.array
        orig_shape = array.shape
        if filter_ is not None:
            indexes_pixel_to_process = filter_.filter(batch)
            array = array[indexes_pixel_to_process]
        else:
            array = self._flat_array(array)

        prediction = self._classify_array(array, threshold)
        if filter_ is not None:
            res = np.zeros(orig_shape[:2], dtype=prediction.dtype)
            res[indexes_pixel_to_process] = prediction
        else:
            res = prediction.reshape(orig_shape[:2])
        return res

    def _classify_batch_by_patch(self, slide, start, size, level, patch_size,
                                 filter_, threshold):
        batch = self._get_batch(slide, start, size, level)
        array = batch.array
        batch_shape = array.shape
        res_shape = (batch_shape[0] // patch_size[0],
                     batch_shape[1] // patch_size[1])

        res = np.zeros(res_shape, dtype='uint8' if threshold else 'float32')
        patches_to_predict = [
            p for p in batch.get_patches(patch_size, filter_)
            if p.array.shape[:2] == patch_size
        ]
        prediction = self._classify_array(
            np.stack([p.array for p in patches_to_predict]),
            threshold) if len(patches_to_predict) else [[]]
        for i, p in enumerate(prediction[0]):
            patch = patches_to_predict[i]
            res[patch.row, patch.col] = p

        return res

    def _classify_array(self, array, threshold) -> np.ndarray:
        prediction = self.model.predict(array)
        if threshold:
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            prediction = prediction.astype('uint8')
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

    def _flat_array(self, array: np.ndarray) -> np.ndarray:
        n_px = array.shape[0] * array.shape[1]
        array = array[:, :, :3].reshape(n_px, 3)
        return array


@dataclass
class Batch:
    slide: Slide
    location: Tuple[int, int]
    size: Tuple[int, int]
    level: int

    def __post_init__(self):
        self.downsample = self.slide.level_downsamples[self.level]
        self._array = None

    @property
    def row(self):
        return int(self.location[0] // self.downsample)

    @property
    def column(self):
        return int(self.location[1] // self.downsample)

    @property
    def array(self):
        if self._array is None:
            logger.debug('reading region for batch %s', self)
            image = self.slide.read_region(self.location[::-1], self.level,
                                           self.size[::-1])
            self._array = image.to_array(True)
        return self._array

    def get_patches(self,
                    patch_size: Tuple[int, int],
                    filter_: Filter = None) -> "Patch":
        dimensions = self.slide.level_dimensions[self.level][::-1]
        if filter_ is None:
            for row in range(0, self.size[0], patch_size[0]):
                if self.row // self.downsample + row + patch_size[
                        0] > dimensions[0]:
                    continue
                for col in range(0, self.size[1], patch_size[1]):
                    if self.column // self.downsample + col + patch_size[
                            1] > dimensions[1]:
                        continue
                    patch_row = int(row // patch_size[0])
                    patch_column = int(col // patch_size[1])
                    yield Patch(self, patch_row, patch_column, patch_size)

        else:
            index_patches = filter_.filter(self, patch_size)
            for p in np.argwhere(index_patches):
                patch = Patch(self, p[0], p[1], patch_size)
                logger.debug('yielding patch %s', patch)
                yield patch


@dataclass
class BatchIterator:
    slide: Slide
    level: int
    n_batch: int

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
                yield Batch(
                    self.slide,
                    (round(i * self._downsample), round(j * self._downsample)),
                    size, self.level)

    def __len__(self):
        return self._level_dimensions[0] // self._batch_size[0]


@dataclass
class Patch:
    slide: Slide
    row: int
    column: int
    level: int
    size: Tuple[int, int]

    def __post_init__(self):
        self._array = None

    @property
    def array(self):
        if self._array is None:
            location = (self.row * self.size[0], self.column * self.size[1])
            image = self.slide.read_region(location[::-1], self.level,
                                           self.size)
            self._array = image.to_array(True)
        return self._array
