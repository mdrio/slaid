import abc
import math
import re
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np

from slaid.commons import Mask, Patch, Slide, convert_patch
from slaid.models import Model


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
        patch_size = patch_size or slide.level_dimensions[level]
        if patch_filter is not None:
            patch_filter = PatchFilter.create(slide, patch_filter).filter
            assert patch_size is not None
        else:
            patch_filter = (lambda x: True)

        patches_by_row = defaultdict(list)
        for batch in self._get_batches(slide, level, n_batch, patch_size):
            for p in slide.patches(level, patch_size, batch.start, batch.end):
                if patch_filter(p):
                    patch_mask = self._classify_patch(p, batch, threshold)
                else:
                    patch_mask = self._get_zeros(p.size[::-1], dtype='uint8')
                patches_by_row[p.y].append(patch_mask)

        rows = [
            self._concatenate(patches, axis=1)
            for _, patches in sorted(patches_by_row.items())
        ]
        mask = self._concatenate(rows, axis=0)

        return Mask(mask, level, slide.level_downsamples[level])

    def _get_batches(self, slide, level, n_batch, patch_size):
        dimensions = slide.level_dimensions[level]
        total_patch_rows = dimensions[1] / patch_size[1]
        patch_per_batch = math.ceil(total_patch_rows / n_batch)

        batch_size = patch_per_batch * patch_size[1]
        step = round(batch_size * slide.level_downsamples[level])
        for i in range(0, slide.dimensions[1], step):
            image = slide.read_region((0, i), level,
                                      (dimensions[0], batch_size))
            array = image.to_array(True)
            yield Batch((0, i), (slide.dimensions[0], i + step), array)

    def _classify_patch(
        self,
        patch: Patch,
        batch: "Batch",
        threshold: float = 0.8,
    ) -> np.ndarray:
        y = round(patch.y // patch.level_downsample) - batch.start[1]
        x = round(patch.x // patch.level_downsample)
        image_array = batch.array[y:y + patch.size[1], x:x + patch.size[0]]
        image_array = self._flat_array(image_array)
        prediction = self.model.predict(image_array)
        return self._get_mask(prediction, patch.size[::-1], threshold)

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
        mask = prediction.reshape(shape)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        mask = np.array(mask, dtype=np.uint8)
        return mask


class Batch:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int],
                 array: np.ndarray):
        self.start = start
        self.end = end
        self.array = array
