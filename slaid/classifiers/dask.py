import logging
import threading

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import BasicClassifier, BatchIterator
from slaid.commons.dask import Mask

logger = logging.getLogger('dask')


class Classifier(BasicClassifier):
    lock = threading.Lock()

    def _classify_batches(self, batches: BatchIterator,
                          threshold: float) -> Mask:
        predictions = []
        for batch in batches:
            predictions.append(
                da.from_delayed(delayed(self._classify_batch)(batch,
                                                              threshold),
                                batch.size,
                                dtype='float32' if not threshold else 'uint8'))
        return self._concatenate(predictions, axis=0)

    @staticmethod
    def _get_mask(array, level, downsample):
        return Mask(array, level, downsample)

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
