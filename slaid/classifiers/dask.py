import logging
from typing import Tuple

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier
from slaid.commons import Slide

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
        mask = super().classify(slide, patch_filter, threshold, level,
                                patch_size)
        mask.array = mask.array.compute()
        return mask

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
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return da.concatenate(seq, axis)
