import abc
import logging
import re
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np

from slaid.commons import Mask, Patch, Slide, convert_patch
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


class PatchFilter:
    def __init__(self, slide: Slide, mask: Mask, operator: str, value: float):
        self.slide = slide,
        self.mask = mask
        self.operator = operator
        self.value = value

    @staticmethod
    def create(slide: Slide, condition: str) -> Tuple[str, Callable, float]:
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
        return PatchFilter(slide, mask, operator, value)

    def filter(self, patch: Patch):
        patch = convert_patch(patch, self.slide, self.mask.level_downsample)
        filtered = getattr(self.mask.ratio(patch), self.operator)(self.value)
        return filtered


class BasicClassifier(Classifier):
    def __init__(self, model: "Model", feature: str):
        self.model = model
        self.feature = feature

    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size: Tuple[int, int] = None,
                 n_batch: int = 1) -> Mask:
        if patch_filter is not None:
            patch_filter = PatchFilter.create(slide, patch_filter).filter
            assert patch_size is not None
        else:
            patch_filter = (lambda x: True)

        rows = []
        for i, (start, size) in enumerate(
                self._get_batch_coordinates(slide, level, n_batch,
                                            patch_size)):
            logger.debug('batch %s of %s', i, n_batch)
            rows.append(
                self._classify_batch(slide, start, size, level, patch_size,
                                     patch_filter, threshold))
        mask = self._concatenate(rows, axis=0)

        return Mask(mask, level, slide.level_downsamples[level])

    def _classify_batch(self, slide, start, size, level, patch_size,
                        patch_filter, threshold):
        rows = []
        batch = self._get_batch(slide, start, size, level)
        patches_by_row = defaultdict(list)
        for p in batch.get_patches(patch_size):
            if patch_filter(p):
                patch_mask = self._classify_patch(p, batch, threshold)
            else:
                patch_mask = self._get_zeros(p.array.shape, dtype='uint8')
            patches_by_row[p.y].append(patch_mask)

        for _, r in sorted(patches_by_row.items()):
            rows.append(np.concatenate([_ for _ in r], axis=1))

        return np.concatenate(rows, axis=0)

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

    def get_patches(self, patch_size) -> Patch:
        patch_size = patch_size or self.array.shape[:2]

        for y in range(0, self.array.shape[1], patch_size[1]):
            for x in range(0, self.array.shape[0], patch_size[0]):
                patch = self.array[x:x + patch_size[0], y:y + patch_size[1]]
                yield Patch(self.start[1] + y, self.start[0] + x,
                            patch.shape[:2][::-1], self.level_downsample,
                            patch)
