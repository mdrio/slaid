# NAPARI LAZY OPENSLIDE
#  Copyright (c) 2020, Trevor Manz
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of napari-lazy-openslide nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import logging
from datetime import datetime as dt

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import BasicClassifier
from slaid.classifiers.base import Batch as BaseBatch
from slaid.classifiers.base import BatchIterator, Filter
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

    @property
    def model(self):
        return load_model(self._model.weight_filename,
                          self._model.gpu) if isinstance(
                              self._model, EddlModel) else self._model

    @staticmethod
    def _get_batch_iterator(slide, level, n_batch, color_type, coords,
                            channel):
        return BatchIterator(slide, level, n_batch, color_type, coords,
                             channel, Batch)

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True,
                 n_patch=25) -> Mask:

        logger.info('classify: %s, %s, %s, %s, %s, %s', slide.filename,
                    filter_, threshold, level, n_batch, round_to_0_100)
        #  batches = self._get_batch_iterator(slide, level, n_batch,
        #                                     self._color_type, self._coords,
        #                                     self._channel)
        if self._patch_size:
            predictions = self._classify_patches(slide, self._patch_size,
                                                 level, filter_, threshold,
                                                 n_patch, round_to_0_100)
        else:
            predictions = self._classify_batches(slide, level, n_batch,
                                                 threshold, round_to_0_100)
        predictions = self._threshold(predictions, threshold)
        predictions = self._round_to_0_100(predictions, round_to_0_100)
        return self._get_mask(predictions, level,
                              slide.level_downsamples[level], dt.now(),
                              round_to_0_100)

    def _classify_batches(self, slide, level, n_batch, threshold,
                          round_to_0_100):
        slide_array = da.from_delayed(
            delayed(slide.to_array)(level),
            shape=list(slide.level_dimensions[level][::-1]) + [4],
            dtype='uint8')
        predictions = []
        n_steps = n_batch
        step = slide_array.shape[0] // n_steps
        logger.debug('step %s, n_steps %s', step, n_steps)
        model = delayed(self.model)
        for i in range(0, n_steps, step):
            area = slide_array[i:i + step, :, :3]
            n_px = area.shape[0] * area.shape[1]
            area_reshaped = area.reshape((n_px, 3))

            prediction = da.from_delayed(model.predict(area_reshaped),
                                         shape=(area_reshaped.shape[0], ),
                                         dtype='float32')
            prediction = prediction.reshape(area.shape[0], area.shape[1])
            predictions.append(prediction)
        return da.concatenate(predictions, 0)

    @staticmethod
    def _round_to_0_100(array: np.ndarray, round_: bool) -> np.ndarray:
        return (array * 100).astype('uint8') if round_ else array

    @staticmethod
    def _threshold(array: np.ndarray, threshold: float) -> np.ndarray:
        if threshold is not None:
            array[array >= threshold] = 1
            array[array < threshold] = 0
            return array.astype('uint8')
        return array

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
        patches_to_predict = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        slide_array = da.from_delayed(
            delayed(slide.to_array)(level),
            shape=list(slide.level_dimensions[level][::-1]) + [4],
            dtype='uint8')

        model = delayed(self.model)
        predictions = []
        patches_to_predict = list(patches_to_predict)
        for i in range(0, len(patches_to_predict), n_patch):
            patches = patches_to_predict[i:i + n_patch]
            input_array = da.stack([
                slide_array[p[0]:p[0] + self._patch_size[0],
                            p[1]:p[1] + self._patch_size[1]] for p in patches
            ])
            predictions.append(
                da.from_delayed(model.predict(input_array),
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
            res[patch[0], patch[1]] = p
        return da.array(res, dtype=dtype)

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        print(seq)
        seq = [el for el in seq if el.size]
        return da.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return da.reshape(array, shape)


def _classify_batch(slide_path: str, model_path: str, level: int):
    pass
