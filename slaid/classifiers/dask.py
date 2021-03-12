import logging
from collections import defaultdict
from datetime import datetime as dt

import dask.array as da
import numpy as np
from dask import delayed
from progress.bar import Bar

from slaid.classifiers.base import BasicClassifier
from slaid.classifiers.base import Batch as BaseBatch
from slaid.classifiers.base import BatchIterator, Filter
from slaid.classifiers.base import Patch as BasePatch
from slaid.commons import Slide
from slaid.commons.dask import Mask
from slaid.models.eddl import Model as EddlModel
from slaid.models.eddl import load_model

logger = logging.getLogger('dask')
fh = logging.FileHandler('/tmp/dask.log')
logger.addHandler(fh)


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
                                                 threshold, round_to_0_100,
                                                 filter_)
        predictions = self._threshold(predictions, threshold)
        predictions = self._round_to_0_100(predictions, round_to_0_100)
        return self._get_mask(predictions, level,
                              slide.level_downsamples[level], dt.now(),
                              round_to_0_100)

    def _classify_batches(self, slide, level, n_batch, threshold,
                          round_to_0_100, filter_):
        slide_array = da.from_delayed(
            delayed(slide.to_array)(level),
            shape=list(slide.level_dimensions[level][::-1]) + [4],
            dtype='uint8')

        model = delayed(self.model)
        if filter_ is not None:
            scale_factor = round(slide.level_downsamples[
                filter_.mask.extraction_level]) // round(
                    slide.level_downsamples[level])

            predictions = []
            area_by_row = defaultdict(list)
            for pixel in filter_:
                pixel = pixel * scale_factor

                area = slide_array[pixel[0]:pixel[0] + scale_factor,
                                   pixel[1]:pixel[1] + scale_factor, :3]
                n_px = area.shape[0] * area.shape[1]
                area_reshaped = area.reshape((n_px, 3))
                area_by_row[pixel[0]].append((pixel[1], area_reshaped))

            for row in range(0, slide_array.shape[0], scale_factor):
                step = scale_factor if row + scale_factor < slide_array.shape[
                    0] else slide_array.shape[0] - row

                row_predictions = []
                if row not in area_by_row:
                    row_predictions = da.zeros((step, slide_array.shape[1]),
                                               dtype='float32')
                else:
                    areas = area_by_row[row]
                    prev_y = 0
                    for y, area in areas:
                        pad_0 = y - prev_y
                        if pad_0 > 0:
                            row_predictions.append(
                                da.zeros((scale_factor, pad_0),
                                         dtype='float32'))

                        prediction = da.from_delayed(model.predict(area),
                                                     shape=(area.shape[0], ),
                                                     dtype='float32')
                        row_predictions.append(
                            prediction.reshape(scale_factor, scale_factor))
                        prev_y = y + scale_factor

                    y = slide_array.shape[1]
                    pad_0 = y - prev_y
                    if pad_0:
                        row_predictions.append(
                            da.zeros((scale_factor, pad_0), dtype='float32'))

                    row_predictions = da.concatenate(row_predictions, 1)

                predictions.append(row_predictions)
            predictions = da.concatenate(predictions, 0)
            logger.debug('predictions shape %s', predictions.shape)
            return da.rechunk(predictions)

            #  res = da.concatenate(res)
            #  logger.debug(
            #      'dif shapes %s',
            #      res.shape[0] - slide_array.shape[0] * slide_array.shape[1])
            #  assert res.shape[0] == slide_array.shape[0] * slide_array.shape[1]
            #  logger.debug('res shape %s, slide_array shape %s', res.shape,
            #               slide_array.shape)
            #  return res.reshape((slide_array.shape[0], slide_array.shape[1]))

        else:
            predictions = []
            step = slide_array.shape[0] // n_batch
            logger.debug('n_batch %s, step %s', n_batch, step)
            with Bar('batches', max=n_batch) as bar:
                for i in range(0, slide_array.shape[0], step):
                    bar.next()
                    area = slide_array[i:i + step, :, :3]
                    n_px = area.shape[0] * area.shape[1]
                    area_reshaped = area.reshape((n_px, 3))

                    prediction = da.from_delayed(
                        model.predict(area_reshaped),
                        shape=(area_reshaped.shape[0], ),
                        dtype='float32')
                    prediction = prediction.reshape(area.shape[0],
                                                    area.shape[1])
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
                            p[1]:p[1] + self._patch_size[1], :3]
                for p in patches
            ])
            predictions.append(
                da.from_delayed(model.predict(input_array),
                                shape=(len(patches), ),
                                dtype=dtype))
        if predictions:
            predictions = da.concatenate(predictions)

        predictions = predictions.compute()
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype='float32')
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch[0], patch[1]] = p
        return da.array(res, dtype='float32')

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
