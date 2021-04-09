import logging

import dask.array as da
import numpy as np

from slaid.classifiers.base import BasicClassifier, Filter
from slaid.commons import BasicSlide, Slide
from slaid.commons.dask import Mask
from slaid.models.base import Model

logger = logging.getLogger('dask')
logger.setLevel(logging.DEBUG)


class Classifier(BasicClassifier):
    MASK_CLASS = Mask

    def __init__(self, model: Model, feature: str, compute_mask: bool = False):
        super().__init__(model, feature)
        self.compute_mask = compute_mask

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 round_to_0_100: bool = True,
                 max_MB_prediction=None) -> Mask:
        mask = super().classify(slide, filter_, threshold, level,
                                round_to_0_100, max_MB_prediction)
        if self.compute_mask:
            mask.compute()
        return mask

    def _predict_batch(self, array):
        n_px = array.shape[0] * array.shape[1]
        p = self._model.predict(array.reshape(
            (n_px, 3))).reshape(array.shape[:2])
        return p

    def _classify_batches_no_filter(self, slide_array, max_MB_prediction):
        prediction = slide_array.array.map_blocks(self._predict_batch,
                                                  meta=np.array(
                                                      (), dtype='float32'),
                                                  drop_axis=2)
        return prediction

    def _classify_batches_with_filter(self, slide, slide_array, level, filter_,
                                      max_MB_prediction):

        size = slide_array.size
        logger.debug('filter_.array.shape %s', filter_.array.shape)
        logger.debug('slide_array.array.shape %s', slide_array.array.shape)
        logger.debug('slide_array.array.chunksize %s',
                     slide_array.array.chunksize)
        logger.debug('slide_array.array.numblocks %s',
                     slide_array.array.numblocks)
        scale = (size[0] // filter_.array.shape[0],
                 size[1] // filter_.array.shape[1])
        logger.debug('scale %s', scale)
        filter_array = np.expand_dims(filter_.array, 2)
        filter_array = da.from_array(
            filter_array,
            chunks=(round(filter_array.shape[0] /
                          slide_array.array.numblocks[0]),
                    round(filter_array.shape[1] /
                          slide_array.array.numblocks[1]),
                    filter_array.shape[2]))
        logger.debug('filter_array.numblocks %s chunks %s',
                     filter_array.numblocks, filter_array.chunksize)
        logger.debug('slide_array.array.numblocks %s, chunks %s',
                     slide_array.array.numblocks, slide_array.array.chunksize)
        #  assert filter_array.numblocks[:2] == slide_array.array.numblocks[:2]
        filter_array = da.map_blocks(
            _rescale,
            filter_array,
            dtype='uint8',
            meta=np.array((), dtype='uint8'),
            #  chunks=(slide_array.array.shape[0] //
            #          slide_array.array.numblocks[0],
            #          slide_array.array.shape[1] //
            #          slide_array.array.numblocks[1], 1),
            chunks=(scale[0] * filter_array.chunksize[0],
                    scale[1] * filter_array.chunksize[1], 1),
            scale=scale)
        filter_array = filter_array[:slide_array.size[0], :slide_array.
                                    size[1], :]
        filter_array = filter_array.rechunk(slide_array.array.chunks[:2] +
                                            ((1, )))
        #  filter_.rescale(slide_array.size)
        #  filter_array = np.expand_dims(filter_.array, 2)
        #  filter_array = da.from_array(filter_array,
        #                               chunks=(slide_array.array.chunks[0],
        #                                       slide_array.array.chunks[1], 1))

        logger.debug('filter_array.shape %s', filter_array.shape)
        logger.debug('slide_array.array.shape %s', slide_array.array.shape)
        logger.debug('filter_array.chunksize %s', filter_array.chunksize)
        logger.debug('slide_array.array.chunksize %s',
                     slide_array.array.chunksize)
        logger.debug('filter_array.numblocks %s', filter_array.numblocks)
        logger.debug('slide_array.array.numblocks %s',
                     slide_array.array.numblocks)
        filter_array = filter_array.rechunk(slide_array.array.chunks[:2] +
                                            (1, ))
        prediction = da.map_blocks(self._predict_with_filter,
                                   slide_array.array,
                                   filter_array,
                                   dtype='float32',
                                   meta=np.array((), dtype='float32'),
                                   drop_axis=2)
        return prediction

    def _predict_with_filter(self,
                             array,
                             filter_array,
                             block_id=None,
                             block_info=None):

        logger.debug('block_id %s', block_id)
        logger.debug('block_info %s', block_info)
        res = np.zeros(array.shape[0] * array.shape[1], dtype='float32')

        filter_array = np.squeeze(filter_array)
        logger.debug('filter_array.shape %s', filter_array.shape)
        logger.debug('array.shape %s', array.shape)
        filtered_array = array[filter_array]
        if filtered_array.shape[0] > 0:
            prediction = self._model.predict(filtered_array)
            res[filter_array.reshape(res.shape)] = prediction
        res = res.reshape(array.shape[:2])
        return res

    def _classify_patches(self,
                          slide: BasicSlide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          round_to_0_100: bool = True,
                          max_MB_prediction=None) -> Mask:
        slide_array = self._get_slide_array(slide, level).array
        level_dims = slide_array.shape
        final_shape = [3] + [
            slide_array.shape[i + 1] -
            (slide_array.shape[i + 1] % self._patch_size[i]) for i in range(2)
        ]
        slide_array = slide_array[:, :final_shape[1], :final_shape[2]]
        if filter_ is None:
            filter_array = np.ones(
                (slide_array.shape[1] // self._patch_size[0],
                 slide_array.shape[2] // self._patch_size[1]),
                dtype='bool')
        else:
            filter_array = filter_.array

        if (filter_array == 1).any():
            patches = []
            for x in range(0, filter_array.shape[0]):
                for y in range(0, filter_array.shape[1]):
                    if filter_array[x, y]:
                        patches.append(slide_array[:,
                                                   x * self._patch_size[0]:x *
                                                   self._patch_size[0] +
                                                   self._patch_size[0],
                                                   y * self._patch_size[1]:y *
                                                   self._patch_size[1] +
                                                   self._patch_size[1]])

            slide_array_flat = da.stack(patches)

            slide_array_flat = slide_array_flat.rechunk(
                50, 3, self._patch_size[0], self._patch_size[1])
            if slide_array_flat.shape[0] == 0:
                predictions = np.zeros((level_dims[1] // self._patch_size[0],
                                        level_dims[2] // self._patch_size[1]),
                                       dtype='float32')
            else:
                filtered_predictions = da.map_blocks(self._predict_patches,
                                                     slide_array_flat,
                                                     drop_axis=[1, 2, 3],
                                                     meta=np.array(
                                                         [], dtype='float32'),
                                                     chunks=(1, ))
                predictions = np.zeros((level_dims[1] // self._patch_size[0],
                                        level_dims[2] // self._patch_size[1]),
                                       dtype='float32')
                filtered_predictions = filtered_predictions.compute().flatten()
                #  raise Exception(filter_.array.shape,
                #                  predictions[filter_.array].shape,
                #                  filtered_predictions.shape)

                predictions[filter_array] = filtered_predictions
        else:
            predictions = np.zeros(filter_array.shape, dtype='float32')
        return predictions

    def _predict_patches(self, chunk, block_info=None):
        return self._model.predict(chunk)

    def _predict_patches_old(self, chunk, filter_=None, block_info=None):
        loc = block_info[0]['array-location']
        chunk_shape = [3] + [
            chunk.shape[i + 1] - (chunk.shape[i + 1] % self._patch_size[i])
            for i in range(2)
        ]
        chunk = chunk[:, :chunk_shape[1], :chunk_shape[2]]
        tmp_splits = np.split(chunk,
                              chunk.shape[1] // self._patch_size[0],
                              axis=1)
        splits = []
        for split in tmp_splits:
            splits.extend(
                np.split(split, chunk.shape[2] // self._patch_size[1], axis=2))
        chunk_reshaped = np.array(splits)
        if filter_ is not None:
            scaled_loc = [[
                loc[1][0] // self._patch_size[0],
                loc[1][1] // self._patch_size[1],
            ],
                          [
                              loc[2][0] // self._patch_size[0],
                              loc[2][1] // self._patch_size[1]
                          ]]
            filter_array = filter_.array[
                scaled_loc[0][0]:scaled_loc[0][1],
                scaled_loc[1][0]:scaled_loc[1][1]].reshape(
                    chunk_reshaped.shape[0])
            res = np.zeros(filter_array.shape, dtype='float32')
            chunk_reshaped = chunk_reshaped[filter_array, ...]
            if chunk_reshaped.shape[0] > 0:
                predictions = self._model.predict(chunk_reshaped)
            else:
                predictions = np.empty((), dtype='float32')
            res[filter_array] = predictions
        else:
            res = self._model.predict(chunk_reshaped)

        res = res.reshape(chunk.shape[1] // self._patch_size[0],
                          chunk.shape[2] // self._patch_size[1])
        return res

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        #  seq = [el for el in seq if el.size]
        return da.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return da.reshape(array, shape)


def _rescale(array, scale, block_info=None):
    #  logger.debug('scale %s', scale)
    #  logger.debug('array.shape %s', array.shape)
    res = np.zeros((array.shape[0] * scale[0], array.shape[1] * scale[1], 1),
                   dtype='bool')
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            res[x * scale[0]:x * scale[0] + scale[0],
                y * scale[1]:y * scale[1] + scale[1]] = array[x, y]
    #  res = np.expand_dims(res, 2)
    #  logger.debug('res.shape %s', res.shape)
    return res
