import logging
import threading

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import BasicClassifier
from slaid.classifiers.base import Batch as BaseBatch
from slaid.classifiers.base import BatchIterator
from slaid.classifiers.base import Filter, Model
from slaid.classifiers.base import Patch as BasePatch
from slaid.commons import Slide
from slaid.commons.dask import Mask
from slaid.models.eddl import Model as EddlModel
from slaid.models.eddl import load_model

logger = logging.getLogger('dask')


class Patch(BasePatch):
    @property
    def array(self):
        return delayed(super().array)


class Batch(BaseBatch):
    @property
    def array(self):
        return delayed(super().array)


class Classifier(BasicClassifier):
    MASK_CLASS = Mask

    #  lock = threading.Lock()

    def __init__(self, model: Model, feature: str):
        super().__init__(model, feature)
        self.model = delayed(load_model)(
            model.weight_filename, model.gpu) if isinstance(
                model, EddlModel) else delayed(lambda: model)()

    @staticmethod
    def _get_batch_iterator(slide, level, n_batch, color_type, coords,
                            channel):
        return BatchIterator(slide, level, n_batch, color_type, coords,
                             channel, Batch)

    def _classify_batches(self, batches: BatchIterator, threshold: float,
                          round_to_0_100: bool) -> Mask:
        predictions = []
        c = 0
        for batch in batches:
            logger.debug('batch %s of %s', c, batches.n_batch)
            logger.debug('batch size %s', batch.size)
            predictions.append(
                da.from_delayed(self._classify_batch(batch, threshold,
                                                     round_to_0_100),
                                batch.size,
                                dtype='uint8'
                                if threshold or round_to_0_100 else 'float32'))
            c += 1
        return self._concatenate(predictions, axis=0)

    def _classify_patches(self,
                          slide: Slide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25,
                          round_to_0_100: bool = True) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
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
                da.from_delayed(p.array,
                                shape=(p.size[0], p.size[1], 3),
                                dtype=dtype) for p in patches
            ])
            predictions.append(
                da.from_delayed(self._classify_array(input_array, threshold,
                                                     round_to_0_100),
                                shape=(len(patches), ),
                                dtype=dtype))
        if predictions:
            predictions = da.concatenate(predictions)

        logger.debug('predictions %s', predictions)
        predictions = predictions.compute()
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch.row, patch.column] = p
        return da.array(res, dtype=dtype)

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

    #  def _classify_array(self, array, threshold, round_to_0_100) -> np.ndarray:
    #      with self.lock:
    #          print('locked classify array')
    #          return super()._classify_array(array, threshold, round_to_0_100)
