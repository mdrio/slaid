import abc
import logging
import re
from typing import Tuple

import numpy as np
from scipy import ndimage

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
        if self.mask.level_downsample != batch.downsample:
            mask = ndimage.zoom(mask,
                                self.mask.level_downsample / batch.downsample)

        mask = mask[batch.location[0]:batch.location[0] + batch.array.shape[0],
                    batch.location[1]:batch.location[1] + batch.array.shape[1]]

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
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_zero: float = 0.01) -> Mask:

        patch_size = self.model.patch_size
        batches = []
        for i, (start, size) in enumerate(
                self._get_batch_coordinates(slide, level, n_batch,
                                            patch_size)):
            logger.debug('batch %s of %s', i, n_batch)
            prediction = self._classify_batch(slide, start, size, level,
                                              patch_size, filter_, threshold)
            if prediction.size:
                batches.append(prediction)

        final_dimensions = slide.level_dimensions[
            level][::-1] if not patch_size else (
                slide.level_dimensions[level][1] // patch_size[0],
                slide.level_dimensions[level][0] // patch_size[1])
        mask = self._reshape(self._concatenate(batches, axis=1),
                             final_dimensions)

        mask = self._round_to_zero(mask, round_to_zero)
        return Mask(mask, level, slide.level_downsamples[level])

    def _round_to_zero(self, mask, threshold):
        if threshold > 0:
            mask[mask <= threshold] = 0
        return mask

    def _classify_batch(self, slide, start, size, level, patch_size, filter_,
                        threshold):
        if patch_size:
            return self._classify_batch_by_patch(slide, start, size, level,
                                                 patch_size, filter_,
                                                 threshold)
        batch = self._get_batch(slide, start, size, level)
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

    def _get_batch_coordinates(self, slide, level, n_batch, patch_size):
        dimensions = slide.level_dimensions[level][::-1]
        dimensions_0 = slide.level_dimensions[0][::-1]
        downsample = slide.level_downsamples[level]

        if patch_size is None:
            batch_size = (dimensions[0] // n_batch, dimensions[1])
        else:
            batch_size_0 = patch_size[0]
            div = (dimensions[0] * dimensions[1] //
                   (n_batch * patch_size[0])) // patch_size[1]
            batch_size_1 = min(dimensions[1], div * patch_size[1])
            batch_size = (batch_size_0, batch_size_1)

        step_1 = round(batch_size[1] * downsample)
        step_0 = round(batch_size[0] * downsample)
        for i in range(0, dimensions[0], batch_size[0]):
            size_0 = min(batch_size[0], dimensions[0] - i)
            for j in range(0, dimensions[1], batch_size[1]):

                size = (size_0, min(batch_size[1], dimensions[1] - j))
                yield ((round(i * downsample), round(j * downsample)), size)

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


class Patch:
    def __init__(self,
                 row: int,
                 column: int,
                 size: Tuple[int, int],
                 downsample: float,
                 array: np.ndarray = None):
        self.row = row
        self.col = column
        self.size = size
        self.downsample = downsample
        self.array = array


class Batch:
    def __init__(self, location: Tuple[int, int], size: Tuple[int, int],
                 array: np.ndarray, downsample: float):
        self.location = location
        self.size = size
        self.array = array
        self.downsample = downsample

    def get_patches(self,
                    patch_size: Tuple[int, int],
                    filter_: Filter = None) -> Patch:
        patch_size = patch_size or self.array.shape[:2]
        if filter_ is None:
            for row in range(0, self.array.shape[0], patch_size[0]):
                for col in range(0, self.array.shape[1], patch_size[1]):
                    array = self.array[row:row + patch_size[0],
                                       col:col + patch_size[1]]
                    yield Patch(row // patch_size[0], col // patch_size[1],
                                array.shape, self.downsample, array)

        else:
            index_patches = filter_.filter(self, patch_size)
            for p in np.argwhere(index_patches):
                p = p * patch_size
                array = self.array[p[0]:p[0] + patch_size[0],
                                   p[1]:p[1] + patch_size[1]]
                yield Patch(p[0] // patch_size[0], p[1] // patch_size[1],
                            array.shape, self.downsample, array)
