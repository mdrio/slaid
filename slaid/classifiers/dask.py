import logging
import threading
from collections import defaultdict

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import (BasicClassifier, BatchIterator, Filter,
                                    Patch)
from slaid.commons import Slide
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

    def _classify_patches(self,
                          slide: Slide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold else 'float32'

        patch_indexes = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        patches_to_predict = [
            Patch(slide, p[0], p[1], level, patch_size) for p in patch_indexes
        ]

        # adding fake patches, workaround for
        # https://github.com/deephealthproject/eddl/issues/236
        #  for _ in range(patch_to_add):
        #      patches_to_predict.append(patches_to_predict[0])

        predictions = []
        for i in range(0, len(patches_to_predict), n_patch):
            patches = patches_to_predict[i:i + n_patch]
            input_array = da.stack([
                da.from_delayed(delayed(getattr)(p, 'array'),
                                shape=(p.size[0], p.size[1], 3),
                                dtype=dtype) for p in patches
            ])
            predictions.append(
                da.from_delayed(delayed(self._classify_array)(input_array,
                                                              threshold),
                                shape=(len(patches), ),
                                dtype=dtype))
        if predictions:
            predictions = da.concatenate(predictions)
        predictions_per_row = defaultdict(lambda: defaultdict(list))
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            predictions_per_row[patch.row][patch.column] = p

        rows_counter = 0
        rows = []
        row_size = dimensions[0] // patch_size[0]
        col_size = dimensions[1] // patch_size[1]
        for i in sorted(predictions_per_row.keys()):
            if i - rows_counter > 0:
                rows.append(da.zeros((i - rows_counter, col_size),
                                     dtype=dtype))
            columns_counter = 0
            row = []
            for j in sorted(predictions_per_row[i].keys()):
                if j - columns_counter > 0:
                    row.append(da.zeros(j - columns_counter, dtype=dtype))
                row.append(da.array([predictions_per_row[i][j]]))
                columns_counter = j + 1
            if col_size - columns_counter > 0:
                row.append(da.zeros(col_size - columns_counter, dtype=dtype))
            rows.append(da.concatenate(row).reshape(1, col_size))
            rows_counter = i + 1
        if row_size - rows_counter > 0:
            rows.append(
                da.zeros((row_size - rows_counter, col_size), dtype=dtype))

        return da.concatenate(rows).reshape(row_size, col_size).rechunk()

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
