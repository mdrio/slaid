from collections import defaultdict
import abc
import re
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from slaid.commons import Image, Mask, Patch, Slide, convert_patch
from slaid.models import Model


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None):
        pass


@dataclass
class PatchFilter:
    slide: Slide
    mask: Mask
    operator: str
    value: float

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
        self._model = model
        self._feature = feature

    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size=None):
        if patch_filter is not None:
            patch_filter = PatchFilter.create(slide, patch_filter).filter
            assert patch_size is not None
        else:
            patch_filter = (lambda x: True)

        if patch_size is not None:
            patches_by_row = defaultdict(list)
            for p in slide.patches(level, patch_size):
                if patch_filter(p):
                    patch_mask = self.classify_patch(slide, (p.x, p.y), level,
                                                     p.size)
                else:
                    patch_mask = self._get_zeros(p.size[::-1], dtype='uint8')
                patches_by_row[p.y].append(patch_mask)

            rows = [
                self._concatenate(patches, axis=1)
                for _, patches in sorted(patches_by_row.items())
            ]
            mask = self._concatenate(rows, axis=0)

        else:
            mask = self.classify_patch(slide, (0, 0), level,
                                       slide.level_dimensions[level],
                                       threshold)

        slide.masks[self._feature] = Mask(mask, level,
                                          slide.level_downsamples[level])

    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        image = slide.read_region(location, level, size)
        image_array = self._get_image_array(image)
        prediction = self._model.predict(image_array)
        return self._get_mask(prediction, size[::-1], threshold)

    @staticmethod
    def _get_zeros(size, dtype):
        return np.zeros(size, dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)

    def _get_image_array(self, image: Image) -> np.ndarray:
        image_array = image.to_array(True)
        n_px = image_array.shape[0] * image_array.shape[1]
        image_array = image_array[:, :, :3].reshape(n_px, 3)
        return image_array

    def _get_mask(self, prediction: np.ndarray, shape: Tuple[int, int],
                  threshold: float) -> np.ndarray:
        mask = prediction.reshape(shape)
        mask[mask < threshold] = 0
        mask[mask > threshold] = 1
        mask = np.array(mask, dtype=np.uint8)
        return mask
