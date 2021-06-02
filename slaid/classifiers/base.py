import abc
import logging
from datetime import datetime as dt
from typing import Tuple

import numpy as np
from progress.bar import Bar
from skimage.util import view_as_blocks

from slaid.commons import BasicSlide, Mask, Slide
from slaid.commons.base import Filter, ImageInfo
from slaid.models import Model

logger = logging.getLogger('classify')
fh = logging.FileHandler('/tmp/base-classifier.log')
logger.addHandler(fh)


class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(self,
                 slide: BasicSlide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 n_batch: int = 1,
                 round_to_0_100: bool = True,
                 region: Tuple[int, int, int, int] = None) -> Mask:
        pass


class BasicClassifier(Classifier):
    MASK_CLASS = Mask

    def __init__(self, model: "Model", feature: str):
        self._model = model
        self.feature = feature
        try:
            self._patch_size = self._model.patch_size
            self._image_info = model.image_info
        except AttributeError as ex:
            logger.error(ex)
            raise ex
            self._patch_size = None
            self._image_info = ImageInfo(ImageInfo.COLORTYPE('rgb'),
                                         ImageInfo.COORD('yx'),
                                         ImageInfo.CHANNEL('last'))

    @property
    def model(self):
        return self._model

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 round_to_0_100: bool = True,
                 dest_array=None,
                 chunk: Tuple[int, int] = None) -> Mask:

        slide_array = self._get_slide_array(slide, level)
        filter_ = filter_.rescale(slide_array.size) if filter_ else np.ones(
            slide_array.size, dtype='bool')
        chunk = chunk or slide_array.size
        dest_array = dest_array if dest_array is not None else np.empty(
            (0, slide_array.size[1]), dtype='float32')

        if self._patch_size:
            predictions = self._classify_patches(slide, self._patch_size,
                                                 level, filter_, threshold,
                                                 round_to_0_100)
        else:
            dest_array = self._classify(slide_array, filter_, dest_array,
                                        chunk, threshold, round_to_0_100)

        return self._get_mask(dest_array, level,
                              slide.level_downsamples[level], dt.now(),
                              round_to_0_100)

    def _classify(self, slide_array, filter_, dest_array, chunk, threshold,
                  round_to_0_100):
        with Bar('Processing', max=filter_.shape[0] // chunk[0]) as bar:
            for x in range(0, filter_.shape[0], chunk[0]):
                row = np.empty((min(chunk[0], filter_.shape[0] - x), 0),
                               dtype='float32')
                for y in range(0, filter_.shape[1], chunk[1]):
                    block = slide_array[x:x + chunk[0], y:y + chunk[1]]
                    filter_block = filter_[x:x + chunk[0], y:y + chunk[1]]

                    res = np.zeros(filter_block.shape, dtype='float32')
                    to_predict = block[filter_block]
                    prediction = self._predict(to_predict)
                    res[filter_block] = prediction
                    res = self._threshold(res, threshold)
                    res = self._round_to_0_100(res, round_to_0_100)
                    row = self._append(row, res, 1)
                dest_array = self._append(dest_array, row, 0)
                bar.next()

        return dest_array

    @staticmethod
    def _append(array, values, axis):
        import zarr
        if isinstance(array, np.ndarray):
            return np.append(array, values, axis)
        elif isinstance(array, zarr.core.Array):
            array.append(values, axis)
            return array

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

    def _get_mask(self, array, level, downsample, datetime, round_to_0_100):
        return self.MASK_CLASS(array,
                               level,
                               downsample,
                               datetime,
                               round_to_0_100,
                               model=str(self._model))

    def _classify_patches(self,
                          slide: BasicSlide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          round_to_0_100: bool = True,
                          max_MB_prediction=None) -> Mask:
        raise NotImplementedError()

    def _classify_batches(self, slide, level, filter_, max_MB_prediction):
        slide_array = self._get_slide_array(slide, level)
        if filter_ is None:
            return self._classify_batches_no_filter(slide_array,
                                                    max_MB_prediction)

        else:
            return self._classify_batches_with_filter(slide, slide_array,
                                                      level, filter_,
                                                      max_MB_prediction)

    def _get_slide_array(self, slide, level):
        return slide[level].convert(self.model.image_info)

    def _classify_batches_with_filter(self, slide, slide_array, level, filter_,
                                      max_MB_prediction):

        filter_.rescale(slide_array.size)
        res = np.zeros(slide_array.size[0] * slide_array.size[1],
                       dtype='float32')
        prediction = self._model.predict(slide_array.array[filter_.array])
        res[filter_.array.reshape(res.shape)] = prediction
        res = res.reshape(slide_array.size)
        return res

    def _classify_batches_no_filter(self, slide_array, max_MB_prediction):
        predictions = []
        step = slide_array.size[0] if max_MB_prediction is None else round(
            max_MB_prediction * 10**6 // (3 * slide_array.size[1]))
        logger.info('max_MB_prediction %s, size_1 %s step %s, n_batch %s',
                    max_MB_prediction, slide_array.size[1], step,
                    slide_array.size[1] // step)
        for i in range(0, slide_array.size[0], step):
            area = slide_array[i:i + step, :]
            n_px = area.size[0] * area.size[1]
            area_reshaped = area.reshape((n_px, ))

            prediction = self._predict(area_reshaped)

            prediction = prediction.reshape(area.size[0], area.size[1])
            predictions.append(prediction)
        return self._concatenate(predictions, 0)

    def _predict(self, area):
        return self.model.predict(area.array)

    def _classify_batch(self, batch, threshold, round_to_0_100):
        logger.debug('classify batch %s', batch)
        # FIXME
        filter_ = None
        array = batch.array()
        logger.debug('get array')
        orig_shape = batch.shape
        if filter_ is not None:
            indexes_pixel_to_process = filter_.filter(batch)
            array = array[indexes_pixel_to_process]
        else:
            logger.debug('flattening batch')
            array = batch.flatten()

        logger.debug('start predictions')
        prediction = self._classify_array(array, threshold, round_to_0_100)
        #      self._classify_array(_, threshold, round_to_0_100)
        #  prediction = np.concatenate([
        #      self._classify_array(_, threshold, round_to_0_100)
        #      for _ in np.array_split(array, 10000)
        #  ])
        #  logger.debug('end predictions')
        #      self._classify_array(_, threshold, round_to_0_100)
        if filter_ is not None:
            res = np.zeros(orig_shape[:2], dtype=prediction.dtype)
            res[indexes_pixel_to_process] = prediction
        else:
            res = prediction.reshape(orig_shape[:2])
        return res

    def _classify_array(self, array, threshold, round_to_0_100) -> np.ndarray:
        logger.debug('classify array')
        prediction = self.model.predict(array)
        if round_to_0_100:
            prediction = prediction * 100
            return prediction.astype('uint8')
        if threshold:
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            return prediction.astype('uint8')

        return prediction.astype('float32')

    @staticmethod
    def _concatenate(seq, axis):
        return np.concatenate(seq, axis)
