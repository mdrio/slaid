import logging
import threading
from typing import Tuple

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import BasicClassifier
from slaid.commons import Slide
from slaid.commons.dask import Mask
from collections import defaultdict

logger = logging.getLogger('dask')


class Classifier(BasicClassifier):
    lock = threading.Lock()

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_zero: float = 0.01) -> Mask:
        batches = defaultdict(list)
        try:
            patch_size = self.model.patch_size
        except:
            patch_size = None
        for i, (start, size) in enumerate(
                self._get_batch_coordinates(slide, level, n_batch,
                                            patch_size)):
            batches[start[0]].append(
                da.from_delayed(
                    delayed(self._classify_batch)(slide, start, size, level,
                                                  patch_size, filter_,
                                                  threshold),
                    shape=size if not patch_size else tuple(
                        [size[i] // patch_size[i] for i in range(2)]),
                    dtype='uint8' if threshold else 'float32'))
        for k, v in batches.items():
            batches[k] = da.concatenate(v, 1)
        # FIXME code duplication with BaseClassifier
        final_dimensions = slide.level_dimensions[
            level][::-1] if not patch_size else (
                slide.level_dimensions[level][1] // patch_size[0],
                slide.level_dimensions[level][0] // patch_size[1])

        mask = self._reshape(self._concatenate(batches.values(), axis=0),
                             final_dimensions)
        mask = self._round_to_zero(mask, round_to_zero)
        return Mask(mask, level, slide.level_downsamples[level])

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        seq = [el for el in seq if el.size]
        return da.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return da.reshape(array, shape)

    def _classify_array(self, array, threshold) -> np.ndarray:
        with self.lock:
            return super()._classify_array(array, threshold)
