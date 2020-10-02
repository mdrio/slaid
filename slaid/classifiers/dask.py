from collections import defaultdict
import logging
from typing import Tuple

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, PatchFilter
from slaid.commons import Mask, Slide

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    return Client(*args, **kwargs)


class Classifier(BasicClassifier):
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
        slide.masks[self._feature].array = slide.masks[
            self._feature].array.compute()

    def classify_patch(
        self,
        slide,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        threshold: float = 0.8,
    ) -> np.ndarray:
        return da.from_delayed(
            self._classify_patch(slide, location, level, size), size[::-1],
            'uint8')

    @delayed
    def _classify_patch(
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
        return da.zeros(size, dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return da.concatenate(seq, axis)
