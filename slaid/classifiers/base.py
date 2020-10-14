import abc
import logging
import re
from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy import ndimage

from slaid.commons import Mask, Patch, Slide
from slaid.models import Model

logger = logging.getLogger('classify')


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None) -> Mask:
        pass


class Filter:
    def __init__(self, mask: Mask, operator: str, value: float):
        self.operator = operator
        self.value = value
        self.mask = mask

    @staticmethod
    def create(slide: Slide, condition: str) -> "Filter":
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
        return Filter(mask, operator, value)

    def filter(self,
               batch: "Batch",
               patch_size: Tuple[int, int] = None) -> np.ndarray:
        mask = self.mask.array
        if self.mask.level_downsample != batch.level_downsample:
            mask = ndimage.zoom(
                mask, self.mask.level_downsample / batch.level_downsample)

        mask = mask[batch.start[0]:batch.start[0] + batch.array.shape[0],
                    batch.start[1]:batch.start[1] + batch.array.shape[1]]

        if patch_size is not None:
            mask = self._compute_mean_patch(mask, patch_size)
        return getattr(mask, self.operator)(self.value)

    def _compute_mean_patch(self, array, patch_size):
        sum_ = np.add.reduceat(np.add.reduceat(array,
                                               np.arange(
                                                   0, array.shape[0],
                                                   patch_size[0]),
                                               axis=0),
                               np.arange(0, array.shape[1], patch_size[1]),
                               axis=1)
        return sum_ / (patch_size[0] * patch_size[1])


class BasicClassifier(Classifier):
    def __init__(self, model: "Model", feature: str):
        self.model = model
        self.feature = feature

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size: Tuple[int, int] = None,
                 n_batch: int = 1) -> Mask:

        rows = []
        for i, (start, size) in enumerate(
                self._get_batch_coordinates(slide, level, n_batch,
                                            patch_size)):
            logger.debug('batch %s of %s', i, n_batch)
            rows.append(
                self._classify_batch(slide, start, size, level, patch_size,
                                     filter_, threshold))
        mask = self._concatenate(rows, axis=0)

        return Mask(mask, level, slide.level_downsamples[level])

    def _classify_batch(self, slide, start, size, level, patch_size, filter_,
                        threshold):
        batch = self._get_batch(slide, start, size, level)
        array = batch.array
        orig_shape = array.shape
        if filter_ is not None:
            indexes_pixel_to_process = filter_.filter(batch)
            array = array[indexes_pixel_to_process]
        else:
            array = self._flat_array(array)

        prediction = self.model.predict(array)
        prediction[prediction > threshold] = 1
        if filter_ is not None:
            res = np.zeros(orig_shape[:2], dtype='uint8')
            res[indexes_pixel_to_process] = prediction
        else:
            res = prediction.reshape(orig_shape[:2])
        return res

    def _get_batch_coordinates(self, slide, level, n_batch, patch_size):
        dimensions = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        batch_size = (dimensions[0], dimensions[1] // n_batch)
        if patch_size is not None:
            batch_size_1 = batch_size[1] - (batch_size[1] % patch_size[1])
            batch_size = (batch_size[0], batch_size_1)

        step = round(batch_size[1] * downsample)
        for i in range(0, slide.dimensions[1], step):
            size = (dimensions[0],
                    min(batch_size[1], dimensions[1] - int(i // downsample)))
            yield ((0, i), size)

    @staticmethod
    def _get_batch(slide, start, size, level):
        image = slide.read_region(start, level, size)
        array = image.to_array(True)
        return Batch(start, size, array, slide.level_downsamples[level])

    def _classify_patch(
        self,
        patch: Patch,
        batch: "Batch",
        threshold: float = 0.8,
    ) -> np.ndarray:
        image_array = patch.array
        orig_shape = image_array.shape[:2]
        image_array = self._flat_array(image_array)
        prediction = self.model.predict(image_array)
        return self._get_mask(prediction, orig_shape, threshold)

    @staticmethod
    def _get_zeros(size, dtype):
        return np.zeros(size, dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)

    def _flat_array(self, array: np.ndarray) -> np.ndarray:
        n_px = array.shape[0] * array.shape[1]
        array = array[:, :, :3].reshape(n_px, 3)
        return array

    def _get_mask(self, prediction: np.ndarray, shape: Tuple[int, int],
                  threshold: float) -> np.ndarray:
        logger.debug('prediction.shape %s, shape %s', prediction.shape, shape)
        mask = prediction.reshape(shape)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        mask = np.array(mask, dtype=np.uint8)
        return mask


class Batch:
    def __init__(self, start: Tuple[int, int], size: Tuple[int, int],
                 array: np.ndarray, level_downsample: float):
        self.start = start
        self.size = size
        self.array = array
        self.level_downsample = level_downsample

    def get_patches(self, patch_size: Tuple[int, int],
                    filter: Filter) -> Patch:
        patch_size = patch_size or self.array.shape[:2]

        for y in range(0, self.array.shape[1], patch_size[1]):
            for x in range(0, self.array.shape[0], patch_size[0]):
                patch = self.array[x:x + patch_size[0], y:y + patch_size[1]]
                yield Patch(self.start[1] + y, self.start[0] + x,
                            patch.shape[:2][::-1], self.level_downsample,
                            patch)
