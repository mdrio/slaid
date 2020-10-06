from typing import Tuple
import logging

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, Batch, Mask
from slaid.commons import Patch, Slide

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)


class Classifier(BasicClassifier):
    def classify(self,
                 slide: Slide,
                 patch_filter=None,
                 threshold: float = 0.8,
                 level: int = 2,
                 patch_size: Tuple[int, int] = None,
                 n_batch: int = 1) -> Mask:

        mask = super().classify(slide, patch_filter, threshold, level,
                                patch_size, n_batch)
        mask.array = mask.array.compute()
        return mask

    def _classify_patch(
        self,
        patch: Patch,
        batch: "Batch",
        threshold: float = 0.8,
    ) -> np.ndarray:
        array = self._classify_patch_delayed(patch, batch, threshold)
        return da.from_delayed(array, patch.size[::-1], 'uint8')

    @delayed
    def _classify_patch_delayed(
        self,
        patch: Patch,
        batch: "Batch",
        threshold: float = 0.8,
    ) -> np.ndarray:
        return super()._classify_patch(patch, batch, threshold)

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return da.concatenate(seq, axis)
