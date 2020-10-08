import logging
from typing import Tuple

import dask.array as da
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, Mask, PatchFilter
from slaid.commons import Slide

logger = logging.getLogger('dask')


def init_client(*args, **kwargs):
    logger.debug('init dask client with %s, %s', args, kwargs)
    return Client(*args, **kwargs)
    #  import dask
    #  dask.config.set(scheduler='synchronous')


class Classifier(BasicClassifier):
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
            #  rows.append(
            #      da.from_array(
            #          self._classify_batch(slide, start, size, level, patch_size,
            #                               patch_filter, threshold)))
            rows.append(
                da.from_delayed(delayed(self._classify_batch)(slide, start,
                                                              size, level,
                                                              patch_size,
                                                              patch_filter,
                                                              threshold),
                                shape=size[::-1],
                                dtype='uint8'))
        print(rows)
        mask = self._concatenate(rows, axis=0)
        return Mask(mask.compute(rerun_exceptions_locally=True), level,
                    slide.level_downsamples[level])

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        return da.concatenate(seq, axis)
