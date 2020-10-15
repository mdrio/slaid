import logging
import threading
from typing import Tuple

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, Batch, Filter, Mask, Patch
from slaid.commons import Slide

logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)
    #  import dask
    #  dask.config.set(scheduler='synchronous')


class Classifier(BasicClassifier):
    lock = threading.Lock()

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
                da.from_delayed(delayed(self._classify_batch)(slide, start,
                                                              size, level,
                                                              patch_size,
                                                              filter_,
                                                              threshold),
                                shape=size[::-1],
                                dtype='uint8'))
        mask = self._concatenate(rows, axis=0)
        return Mask(mask.compute(rerun_exceptions_locally=True), level,
                    slide.level_downsamples[level])

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return da.concatenate(seq, axis)

    def _classify_array(self, array, threshold) -> np.ndarray:
        with self.lock:
            return super()._classify_array(array, threshold)
